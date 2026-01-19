"""
Genesys AudioHook WebSocket protocol serializer for Pipecat.

This serializer implements the Genesys AudioHook protocol for bidirectional
audio streaming between Pipecat pipelines and Genesys Cloud contact center.

Protocol Reference:
- https://developer.genesys.cloud/devapps/audiohook

Audio Format:
- PCMU (μ-law) at 8kHz sample rate (preferred)
- L16 (16-bit linear PCM) at 8kHz also supported
- Mono or Stereo (external on left, internal on right)
"""

import json
import uuid
from datetime import timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.audio.utils import ulaw_to_pcm, pcm_to_ulaw


class AudioHookMessageType(str, Enum):
    """AudioHook protocol message types."""
    OPEN = "open"
    OPENED = "opened"
    CLOSE = "close"
    CLOSED = "closed"
    PAUSE = "pause"
    RESUMED = "resumed"
    PING = "ping"
    PONG = "pong"
    UPDATE = "update"
    EVENT = "event"
    ERROR = "error"
    DISCONNECT = "disconnect"


class AudioHookChannel(str, Enum):
    """AudioHook audio channel configuration."""
    EXTERNAL = "external"  # Customer audio only (mono)
    INTERNAL = "internal"  # Agent audio only (mono)
    BOTH = "both"          # Stereo: external=left, internal=right


class AudioHookMediaFormat(str, Enum):
    """Supported audio formats."""
    PCMU = "PCMU"  # μ-law, 8kHz
    L16 = "L16"    # 16-bit linear PCM, 8kHz


class GenesysAudioHookSerializer(FrameSerializer):
    """Serializer for Genesys AudioHook WebSocket protocol.

    This serializer handles converting between Pipecat frames and Genesys
    AudioHook protocol messages. It supports:
    
    - Bidirectional audio streaming (PCMU at 8kHz)
    - Session lifecycle management (open, close, pause)
    - Probe mode for health checks
    - Custom configuration passthrough
    - Event messaging back to Genesys Cloud

    The AudioHook protocol uses:
    - Text WebSocket frames for JSON control messages
    - Binary WebSocket frames for audio data

    Example usage:
        ```python
        serializer = GenesysAudioHookSerializer(
            session_id="abc-123",
            params=GenesysAudioHookSerializer.InputParams(
                channel=AudioHookChannel.BOTH,
            )
        )
        
        # Use with WebSocket transport
        transport = WebsocketServerTransport(
            serializer=serializer,
            ...
        )
        ```

    Attributes:
        PROTOCOL_VERSION: The AudioHook protocol version (currently "2").
    """

    PROTOCOL_VERSION = "2"

    class InputParams(BaseModel):
        """Configuration parameters for GenesysAudioHookSerializer.

        Parameters:
            genesys_sample_rate: Sample rate used by Genesys (8000 Hz).
            sample_rate: Optional override for pipeline input sample rate.
            channel: Which audio channels to process (external, internal, both).
            media_format: Audio format (PCMU or L16).
            process_external: Whether to process external (customer) audio.
            process_internal: Whether to process internal (agent) audio.
            enable_interruption_events: Send interruption events to Genesys.
        """

        genesys_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        channel: AudioHookChannel = AudioHookChannel.EXTERNAL
        media_format: AudioHookMediaFormat = AudioHookMediaFormat.PCMU
        process_external: bool = True
        process_internal: bool = False
        enable_interruption_events: bool = True

    def __init__(
        self,
        session_id: Optional[str] = None,
        params: Optional[InputParams] = None,
        send_message_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """Initialize the GenesysAudioHookSerializer.

        Args:
            session_id: The AudioHook session ID (received in open message).
            params: Configuration parameters.
            send_message_callback: Optional async callback to send messages directly
                                   (bypassing the pipeline). Used for urgent messages like pong.
        """
        self._params = params or GenesysAudioHookSerializer.InputParams()
        self._session_id = session_id or ""
        self._send_message_callback = send_message_callback
        
        self._genesys_sample_rate = self._params.genesys_sample_rate
        self._sample_rate = 0  # Pipeline input rate, set in setup()
        
        # Use Pipecat's official SOXR resampler
        # Only used for TTS output (16kHz → 8kHz), input goes without resampling
        self._input_resampler = SOXRStreamAudioResampler()
        self._output_resampler = SOXRStreamAudioResampler()
        
        # Protocol state
        self._client_seq = 0
        self._server_seq = 0
        self._is_open = False
        self._is_paused = False
        self._position = timedelta(0)
        
        # TTS output state
        self._tts_chunk_count = 0 
        
        # Session metadata
        self._conversation_id: Optional[str] = None
        self._participant: Optional[Dict[str, Any]] = None
        self._custom_config: Optional[Dict[str, Any]] = None
        self._media_info: Optional[List[Dict[str, Any]]] = None
        self._input_variables: Optional[Dict[str, Any]] = None  # Custom input from Genesys

    def set_send_message_callback(self, callback: Callable[[str], Awaitable[None]]):
        """Set the callback for sending urgent messages directly.
        
        Args:
            callback: An async function that takes a string message and sends it.
        """
        self._send_message_callback = callback

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    @property
    def conversation_id(self) -> Optional[str]:
        """Get the Genesys conversation ID."""
        return self._conversation_id

    @property
    def is_open(self) -> bool:
        """Check if the AudioHook session is open."""
        return self._is_open

    @property
    def is_paused(self) -> bool:
        """Check if audio streaming is paused."""
        return self._is_paused

    @property
    def participant(self) -> Optional[Dict[str, Any]]:
        """Get participant info (ani, dnis, etc.) from the open message."""
        return self._participant

    @property
    def input_variables(self) -> Optional[Dict[str, Any]]:
        """Get custom input variables from the open message."""
        return self._input_variables

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate
        logger.debug(f"GenesysAudioHookSerializer setup with sample_rate={self._sample_rate}")
    
    def reset_tts_state(self):
        """Reset TTS state for a new utterance.
        
        NOTE: We don't reset the resampler because that causes artifacts.
        The resampler maintains its state between utterances for cleaner audio.
        """
        self._tts_chunk_count = 0
    
    def _format_position(self, position: timedelta) -> str:
        """Format a timedelta as ISO 8601 duration string.
        
        Args:
            position: The timedelta to format.
            
        Returns:
            ISO 8601 duration string (e.g., "PT1.5S").
        """
        total_seconds = position.total_seconds()
        return f"PT{total_seconds:.3f}S"

    def _parse_position(self, position_str: str) -> timedelta:
        """Parse an ISO 8601 duration string to timedelta.
        
        Args:
            position_str: ISO 8601 duration string (e.g., "PT1.5S").
            
        Returns:
            Corresponding timedelta.
        """
        # Simple parser for PT#S or PT#.#S format
        if position_str.startswith("PT") and position_str.endswith("S"):
            try:
                seconds = float(position_str[2:-1])
                return timedelta(seconds=seconds)
            except ValueError:
                pass
        return timedelta(0)

    def _next_server_seq(self) -> int:
        """Get the next server sequence number."""
        self._server_seq += 1
        return self._server_seq

    def _create_message(
        self,
        msg_type: AudioHookMessageType,
        parameters: Optional[Dict[str, Any]] = None,
        include_position: bool = True,
    ) -> Dict[str, Any]:
        """Create a protocol message with common fields.
        
        Based on the Genesys AudioHook protocol, responses include:
        - seq: Server's sequence number (incremented per message)
        - clientseq: Echo of the client's last sequence number
        
        Args:
            msg_type: The message type.
            parameters: Optional parameters object.
            include_position: Whether to include position field.
            
        Returns:
            The message dictionary.
        """
        seq = self._next_server_seq()
        msg = {
            "version": self.PROTOCOL_VERSION,
            "type": msg_type.value,
            "seq": seq,
            "clientseq": self._client_seq,
            "id": self._session_id,
        }
        
        if include_position:
            msg["position"] = self._format_position(self._position)
            
        if parameters:
            msg["parameters"] = parameters
            
        return msg

    def create_opened_response(
        self,
        start_paused: bool = False,
        supported_languages: Optional[List[str]] = None,
        selected_language: Optional[str] = None,
    ) -> str:
        """Create an 'opened' response message for the client.
        
        This should be sent in response to an 'open' message from Genesys.
        
        Args:
            start_paused: Whether to start the session paused.
            supported_languages: List of supported language codes.
            selected_language: The selected language code.
            
        Returns:
            JSON string of the opened response message.
        """
        # Build channels list based on configuration
        channels = []
        if self._params.channel == AudioHookChannel.EXTERNAL:
            channels = ["external"]
        elif self._params.channel == AudioHookChannel.INTERNAL:
            channels = ["internal"]
        elif self._params.channel == AudioHookChannel.BOTH:
            channels = ["external", "internal"]
        
        parameters = {
            "startPaused": start_paused,
            "media": [
                {
                    "type": "audio",
                    "format": self._params.media_format.value,
                    "channels": channels,
                    "rate": self._genesys_sample_rate,
                }
            ],
        }
        
        if supported_languages:
            parameters["supportedLanguages"] = supported_languages
        if selected_language:
            parameters["selectedLanguage"] = selected_language
            
        msg = self._create_message(
            AudioHookMessageType.OPENED,
            parameters=parameters,
            include_position=False,  # opened doesn't need position
        )
        
        self._is_open = True
        logger.debug(f"AudioHook session opened: {self._session_id}")
        
        return json.dumps(msg)

    def create_closed_response(self) -> str:
        """Create a 'closed' response message.
        
        This should be sent in response to a 'close' message from Genesys.
        
        Returns:
            JSON string of the closed response message.
        """
        msg = self._create_message(AudioHookMessageType.CLOSED)
        
        self._is_open = False
        logger.debug(f"AudioHook session closed: {self._session_id}")
        
        return json.dumps(msg)

    def create_pong_response(self) -> str:
        """Create a 'pong' response message.
        
        This should be sent in response to a 'ping' message from Genesys.
        
        Returns:
            JSON string of the pong response message.
        """
        msg = self._create_message(AudioHookMessageType.PONG)
        return json.dumps(msg)

    def create_resumed_response(self) -> str:
        """Create a 'resumed' response message.
        
        This should be sent in response to a 'pause' message when ready to resume.
        
        Returns:
            JSON string of the resumed response message.
        """
        msg = self._create_message(AudioHookMessageType.RESUMED)
        
        self._is_paused = False
        logger.debug(f"AudioHook session resumed: {self._session_id}")
        
        return json.dumps(msg)

    def create_event_message(
        self,
        entity_type: str,
        entity_data: Dict[str, Any],
    ) -> str:
        """Create an 'event' message to send data back to Genesys.
        
        This can be used for transcriptions, agent assist, or other events.
        
        Args:
            entity_type: The type of entity (e.g., "transcript").
            entity_data: The entity data.
            
        Returns:
            JSON string of the event message.
        """
        parameters = {
            "entities": [
                {
                    "type": entity_type,
                    **entity_data,
                }
            ]
        }
        
        msg = self._create_message(
            AudioHookMessageType.EVENT,
            parameters=parameters,
        )
        
        return json.dumps(msg)

    def create_disconnect_message(
        self,
        reason: str = "completed",
        action: str = "transfer",
        output_variables: Optional[Dict[str, Any]] = None,
        info: Optional[str] = None,
    ) -> str:
        """Create a 'disconnect' message to initiate session termination.
        
        Args:
            reason: Disconnect reason (e.g., "completed", "error").
            action: Action to take ("transfer" to agent, "finished" if completed).
            output_variables: Custom output variables to pass back to Genesys.
            info: Optional additional information.
            
        Returns:
            JSON string of the disconnect message.
        """
        parameters: Dict[str, Any] = {"reason": reason}
        
        # Build outputVariables
        out_vars = {"action": action}
        if output_variables:
            out_vars.update(output_variables)
        parameters["outputVariables"] = out_vars
        
        if info:
            parameters["info"] = info
            
        msg = self._create_message(
            AudioHookMessageType.DISCONNECT,
            parameters=parameters,
        )
        
        logger.debug(f"AudioHook disconnect: reason={reason}, action={action}")
        return json.dumps(msg)

    def create_error_message(
        self,
        code: int,
        message: str,
        retryable: bool = False,
    ) -> str:
        """Create an 'error' message.
        
        Args:
            code: Error code.
            message: Error message.
            retryable: Whether the operation can be retried.
            
        Returns:
            JSON string of the error message.
        """
        parameters = {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
        
        msg = self._create_message(
            AudioHookMessageType.ERROR,
            parameters=parameters,
        )
        
        logger.error(f"AudioHook error: {code} - {message}")
        return json.dumps(msg)

    def create_barge_in_event(self) -> str:
        """Create a barge-in event message.
        
        This notifies Genesys Cloud that the user has interrupted the bot's
        audio output. Genesys will stop any queued audio playback.
        
        Returns:
            JSON string of the barge-in event message.
        """
        msg = self._create_message(
            AudioHookMessageType.EVENT,
            parameters={
                "entities": [
                    {"type": "barge_in", "data": {}}
                ]
            },
        )
        
        logger.debug("🔇 Barge-in event sent to Genesys")
        
        return json.dumps(msg)

    def create_resume_event(self) -> str:
        """Create a resume event message.
        
        This notifies Genesys that the bot is ready to resume after a barge-in.
        Should be called after the user stops speaking following a barge-in.
        
        Returns:
            JSON string of the resume event message.
        """
        self._barge_in = False
        
        # Note: 'resume' might not be a standard AudioHook message type,
        # but it's used in some implementations
        msg = {
            "version": self.PROTOCOL_VERSION,
            "type": "resume",
            "seq": self._next_server_seq(),
            "clientseq": self._client_seq,
            "id": self._session_id,
            "parameters": {},
        }
        
        logger.debug("Resume event sent")
        return json.dumps(msg)

    def create_interruption_event(
        self,
        reason: str = "user_speaking",
        discarded_bytes: Optional[int] = None,
    ) -> str:
        """Create a generic interruption event message.
        
        This is an alternative to create_barge_in_event() that includes
        more detailed information about the interruption.
        
        Args:
            reason: Reason for interruption (e.g., "user_speaking", "dtmf").
            discarded_bytes: Number of audio bytes discarded due to interruption.
            
        Returns:
            JSON string of the interruption event message.
        """
        entity_data: Dict[str, Any] = {
            "reason": reason,
            "timestamp": self._format_position(self._position),
        }
        
        if discarded_bytes is not None:
            entity_data["discardedAudioBytes"] = discarded_bytes
            
        logger.info(
            f"AudioHook interruption: reason={reason}, "
            f"discarded={discarded_bytes or 0} bytes"
        )
        
        return self.create_event_message(
            entity_type="interruption",
            entity_data=entity_data,
        )

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Genesys AudioHook format.

        Handles conversion of various frame types to AudioHook messages:
        - AudioRawFrame -> Binary audio data
        - EndFrame/CancelFrame -> Disconnect message
        - OutputTransportMessageFrame -> Pass-through JSON

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string (JSON) or bytes (audio), or None.
        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            return self.create_disconnect_message(reason="completed")
            
        elif isinstance(frame, AudioRawFrame):
            if not self._is_open or self._is_paused:
                return None
                
            data = frame.audio
            
            self._tts_chunk_count += 1
            
            # Convert PCM to μ-law at 8kHz for Genesys
            if self._params.media_format == AudioHookMediaFormat.PCMU:
                serialized_data = await pcm_to_ulaw(
                    data,
                    frame.sample_rate,
                    self._genesys_sample_rate,
                    self._output_resampler,
                )
            else:
                # L16 format - just resample if needed
                logger.warning("L16 format not yet fully implemented")
                return None
                
            if serialized_data is None or len(serialized_data) == 0:
                return None
            
            return bytes(serialized_data)
            
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            # Pass through custom JSON messages
            return json.dumps(frame.message)
            
        # Ignore other frames - we don't need to process them here
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Genesys AudioHook data to Pipecat frames.

        Handles:
        - Binary data -> InputAudioRawFrame
        - JSON text -> Protocol messages (open, close, ping, pause, etc.)

        For control messages (open, close, ping, pause), this method handles
        the protocol response internally and logs the events. The application
        should monitor session state via the is_open and is_paused properties.

        Args:
            data: The raw WebSocket data from Genesys.

        Returns:
            A Pipecat frame corresponding to the data, or None if handled internally.
        """
        # Binary data = audio
        if isinstance(data, bytes):
            logger.debug(f"[AUDIO IN] Received {len(data)} bytes from Genesys")
            return await self._deserialize_audio(data)
            
        # Text data = JSON control message
        try:
            message = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AudioHook message: {e}")
            return None
            
        return await self._handle_control_message(message)

    async def _deserialize_audio(self, data: bytes) -> Frame | None:
        """Deserialize binary audio data to an InputAudioRawFrame.
        
        Args:
            data: Raw audio bytes (PCMU or L16).
            
        Returns:
            InputAudioRawFrame with PCM audio at pipeline sample rate.
        """
        if not self._is_open or self._is_paused:
            return None
        
        audio_data = data
        original_len = len(data)
        
        # If Genesys sends stereo audio (BOTH channels), extract only the external channel (left)
        # Stereo audio is interleaved: [L0, R0, L1, R1, ...]
        if self._params.channel == AudioHookChannel.BOTH and len(data) > 0:
            # For PCMU, each sample is 1 byte
            # Extract only bytes at even positions (left channel = external)
            audio_data = bytes(data[i] for i in range(0, len(data), 2))
            logger.debug(f"🔊 Stereo audio: {original_len} bytes → {len(audio_data)} bytes (external channel)")
            
        if self._params.media_format == AudioHookMediaFormat.PCMU:
            # Convert μ-law at 8kHz to PCM at pipeline rate
            deserialized_data = await ulaw_to_pcm(
                audio_data,
                self._genesys_sample_rate,
                self._sample_rate,
                self._input_resampler,
            )
        else:
            # L16 format
            logger.warning("L16 format not yet fully implemented")
            return None
            
        if deserialized_data is None or len(deserialized_data) == 0:
            return None
            
        # Always use mono for STT - ElevenLabs expects single channel
        num_channels = 1
            
        audio_frame = InputAudioRawFrame(
            audio=deserialized_data,
            num_channels=num_channels,
            sample_rate=self._sample_rate,
        )
        
        return audio_frame

    async def _handle_control_message(self, message: Dict[str, Any]) -> Frame | None:
        """Handle a JSON control message from Genesys.
        
        Args:
            message: Parsed JSON message.
            
        Returns:
            Frame if the message should be passed to the pipeline, None otherwise.
        """
        msg_type = message.get("type", "")
        self._client_seq = message.get("seq", 0)
        
        # Update position if provided
        if "position" in message:
            self._position = self._parse_position(message["position"])
        
        if msg_type == AudioHookMessageType.OPEN.value:
            return await self._handle_open(message)
            
        elif msg_type == AudioHookMessageType.CLOSE.value:
            return await self._handle_close(message)
            
        elif msg_type == AudioHookMessageType.PING.value:
            return await self._handle_ping(message)
            
        elif msg_type == AudioHookMessageType.PAUSE.value:
            return await self._handle_pause(message)
            
        elif msg_type == AudioHookMessageType.UPDATE.value:
            return await self._handle_update(message)
            
        elif msg_type == AudioHookMessageType.ERROR.value:
            return await self._handle_error(message)
        
        elif msg_type == "dtmf":
            return await self._handle_dtmf(message)
        
        elif msg_type == "playback_started":
            self._is_playing = True
            logger.debug("Playback started (from Genesys)")
            return None
        
        elif msg_type == "playback_completed":
            self._is_playing = False
            logger.debug("Playback completed (from Genesys)")
            return None
            
        else:
            logger.warning(f"Unknown AudioHook message type: {msg_type}")
            return None

    async def _handle_open(self, message: Dict[str, Any]) -> Frame | None:
        """Handle an 'open' message from Genesys.
        
        This initializes the session with metadata from Genesys Cloud.
        
        Args:
            message: The open message.
            
        Returns:
            None (response should be sent via create_opened_response()).
        """
        self._session_id = message.get("id", str(uuid.uuid4()))
        
        params = message.get("parameters", {})
        self._conversation_id = params.get("conversationId")
        self._participant = params.get("participant")
        self._custom_config = params.get("customConfig")
        self._media_info = params.get("media")  # This is a list of media objects
        self._input_variables = params.get("inputVariables")  # Custom vars from Genesys
        
        # Extract media configuration if present
        # media is a list like: [{"type": "audio", "format": "PCMU", "channels": ["external"], "rate": 8000}]
        media_list = self._media_info
        if media_list and isinstance(media_list, list) and len(media_list) > 0:
            audio_media: Dict[str, Any] = media_list[0]  # Get first media entry
            channels = audio_media.get("channels", [])
            logger.debug(f"📡 Genesys audio config: format={audio_media.get('format')}, channels={channels}, rate={audio_media.get('rate')}")
            # channels is a list like ["external"] or ["external", "internal"]
            if isinstance(channels, list):
                if "external" in channels and "internal" in channels:
                    self._params.channel = AudioHookChannel.BOTH
                    logger.debug("📡 Stereo mode: extracting external channel")
                elif "external" in channels:
                    self._params.channel = AudioHookChannel.EXTERNAL
                    logger.debug("📡 Mono mode: external channel")
                elif "internal" in channels:
                    self._params.channel = AudioHookChannel.INTERNAL
                    logger.debug("📡 Mono mode: internal channel")
        
        # Log participant info for debugging
        ani = self._participant.get("ani", "unknown") if self._participant else "unknown"
        logger.info(
            f"AudioHook open request: session={self._session_id}, "
            f"conversation={self._conversation_id}, ani={ani}"
        )
        
        # Note: Application should call create_opened_response() to respond
        return None

    async def _handle_close(self, message: Dict[str, Any]) -> Frame | None:
        """Handle a 'close' message from Genesys.
        
        Args:
            message: The close message.
            
        Returns:
            EndFrame to signal the pipeline to close.
        """
        params = message.get("parameters", {})
        reason = params.get("reason", "unknown")
        
        logger.info(f"🔴 Genesys closed the connection: {reason}")
        
        self._is_open = False
        
        # Send closed response via callback if available
        if self._send_message_callback:
            try:
                closed_response = self.create_closed_response()
                await self._send_message_callback(closed_response)
            except Exception as e:
                logger.error(f"Failed to send closed response: {e}")
        
        # Return EndFrame to close the pipeline and WebSocket
        return EndFrame()

    async def _handle_ping(self, message: Dict[str, Any]) -> Frame | None:
        """Handle a 'ping' message from Genesys.
        
        Args:
            message: The ping message.
            
        Returns:
            None if pong was sent directly via callback, otherwise
            OutputTransportMessageUrgentFrame with pong response.
        """
        # Create pong response
        pong_response = self.create_pong_response()
        
        # If we have a direct callback, use it for immediate response
        if self._send_message_callback:
            try:
                await self._send_message_callback(pong_response)
                logger.debug("Pong sent directly via callback")
                return None
            except Exception as e:
                logger.error(f"Failed to send pong via callback: {e}")
        
        # Fallback: return as urgent frame to be sent through pipeline
        return OutputTransportMessageUrgentFrame(message=json.loads(pong_response))

    async def _handle_pause(self, message: Dict[str, Any]) -> Frame | None:
        """Handle a 'pause' message from Genesys.
        
        This is used when audio streaming is temporarily suspended
        (e.g., during hold).
        
        Args:
            message: The pause message.
            
        Returns:
            None (response should be sent via create_resumed_response()).
        """
        params = message.get("parameters", {})
        reason = params.get("reason", "unknown")
        
        logger.info(f"AudioHook pause request: reason={reason}")
        
        self._is_paused = True
        
        # Note: Application should call create_resumed_response() when ready
        return None

    async def _handle_update(self, message: Dict[str, Any]) -> Frame | None:
        """Handle an 'update' message from Genesys.
        
        Updates may include changes to participants or configuration.
        
        Args:
            message: The update message.
            
        Returns:
            None.
        """
        params = message.get("parameters", {})
        
        if "participant" in params:
            self._participant = params["participant"]
            
        logger.debug(f"AudioHook update received: {params}")
        
        return None

    async def _handle_error(self, message: Dict[str, Any]) -> Frame | None:
        """Handle an 'error' message from Genesys.
        
        Args:
            message: The error message.
            
        Returns:
            None.
        """
        params = message.get("parameters", {})
        code = params.get("code", 0)
        error_msg = params.get("message", "Unknown error")
        
        logger.error(f"AudioHook error from Genesys: {code} - {error_msg}")
        
        return None

    async def _handle_dtmf(self, message: Dict[str, Any]) -> Frame | None:
        """Handle a 'dtmf' message from Genesys.
        
        DTMF (Dual-Tone Multi-Frequency) events are sent when the user
        presses keys on their phone keypad.
        
        Args:
            message: The DTMF message.
            
        Returns:
            InputDTMFFrame with the pressed digit.
        """
        params = message.get("parameters", {})
        digit = params.get("digit", "")
        
        if not digit:
            logger.warning("DTMF message received without digit")
            return None
        
        logger.info(f"DTMF received: {digit}")
        
        try:
            return InputDTMFFrame(KeypadEntry(digit))
        except ValueError:
            # Invalid digit
            logger.warning(f"Invalid DTMF digit: {digit}")
            return None
