"""Genesys AudioHook Serializer for Pipecat.

This module provides a serializer for integrating Pipecat pipelines with
Genesys Cloud Contact Center via the AudioHook protocol.

Features:
- Bidirectional audio streaming (PCMU μ-law at 8kHz)
- Automatic protocol handshake handling (open/opened, close/closed, ping/pong)
- Input/output variables for Architect flow integration
- DTMF event support
- Barge-in (interruption) events
- Pause/resume support for hold scenarios (optional)

Protocol Reference:
- https://developer.genesys.cloud/devapps/audiohook

Audio Format:
- PCMU (μ-law) at 8kHz sample rate (preferred)
- L16 (16-bit linear PCM) at 8kHz also supported
- Mono (external channel) or Stereo (external on left, internal on right)
"""

import json
import uuid
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.audio.utils import pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


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
    BOTH = "both"  # Stereo: external=left, internal=right


class AudioHookMediaFormat(str, Enum):
    """Supported audio formats."""

    PCMU = "PCMU"  # μ-law, 8kHz
    L16 = "L16"  # 16-bit linear PCM, 8kHz


class GenesysAudioHookSerializer(FrameSerializer):
    """Serializer for Genesys AudioHook WebSocket protocol.

    This serializer handles converting between Pipecat frames and Genesys
    AudioHook protocol messages. It supports:

    - Bidirectional audio streaming (PCMU at 8kHz)
    - Automatic protocol handshake (open/opened, close/closed, ping/pong)
    - Session lifecycle management with pause/resume support
    - Custom input/output variables for Architect flow integration
    - DTMF event handling
    - Barge-in events for interruption support

    The AudioHook protocol uses:
    - Text WebSocket frames for JSON control messages
    - Binary WebSocket frames for audio data

    Example usage:
        ```python
        serializer = GenesysAudioHookSerializer(
            params=GenesysAudioHookSerializer.InputParams(
                channel=AudioHookChannel.EXTERNAL,
                supported_languages=["en-US", "es-ES"],
                selected_language="en-US",
            )
        )

        # Use with FastAPI WebSocket transport
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                serializer=serializer,
                audio_out_fixed_packet_size=1600,  # Important: prevents 429 rate limiting from Genesys
            ),
        )

        # Access call information after connection
        participant = serializer.participant  # ani, dnis, etc.
        input_vars = serializer.input_variables  # Custom vars from Architect

        # Set output variables to return to Architect
        serializer.set_output_variables({"intent": "billing", "resolved": True})
        ```

    Attributes:
        PROTOCOL_VERSION: The AudioHook protocol version (currently "2").
    """

    PROTOCOL_VERSION = "2"

    class InputParams(BaseModel):
        """Configuration parameters for GenesysAudioHookSerializer.

        Attributes:
            genesys_sample_rate: Sample rate used by Genesys (default: 8000 Hz).
            sample_rate: Optional override for pipeline input sample rate.
            channel: Which audio channels to process (external, internal, both).
            media_format: Audio format (PCMU or L16).
            process_external: Whether to process external (customer) audio.
            process_internal: Whether to process internal (agent) audio.
            supported_languages: List of language codes the bot supports (e.g., ["en-US", "es-ES"]).
            selected_language: Default language code to use.
            start_paused: Whether to start the session in paused state.
        """

        genesys_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        channel: AudioHookChannel = AudioHookChannel.EXTERNAL
        media_format: AudioHookMediaFormat = AudioHookMediaFormat.PCMU
        process_external: bool = True
        process_internal: bool = False
        supported_languages: Optional[List[str]] = None
        selected_language: Optional[str] = None
        start_paused: bool = False

    def __init__(
        self,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the GenesysAudioHookSerializer.

        Args:
            params: Configuration parameters.
            **kwargs: Additional arguments passed to BaseObject (e.g., name).
        """
        super().__init__(**kwargs)
        self._params = params or GenesysAudioHookSerializer.InputParams()

        self._genesys_sample_rate = self._params.genesys_sample_rate
        self._sample_rate = 0  # Pipeline input rate, set in setup()
        self._session_id = str(uuid.uuid4())

        # Use Pipecat's official resampler if needed (SOXR)
        # Only used for TTS output (16kHz → 8kHz), input goes without resampling
        self._input_resampler = SOXRStreamAudioResampler()
        self._output_resampler = SOXRStreamAudioResampler()

        # Protocol state
        self._client_seq = 0
        self._server_seq = 0
        self._is_open = False
        self._is_paused = False
        self._position = timedelta(0)

        # Session metadata
        self._conversation_id: Optional[str] = None
        self._participant: Optional[Dict[str, Any]] = None
        self._custom_config: Optional[Dict[str, Any]] = None
        self._media_info: Optional[List[Dict[str, Any]]] = None
        self._input_variables: Optional[Dict[str, Any]] = None  # Custom input from Genesys
        self._output_variables: Optional[Dict[str, Any]] = None  # Custom output to Genesys

        # Event handlers
        self._register_event_handler("on_open")
        self._register_event_handler("on_close")
        self._register_event_handler("on_ping")
        self._register_event_handler("on_pause")
        self._register_event_handler("on_update")
        self._register_event_handler("on_error")
        self._register_event_handler("on_dtmf")

    @property
    def session_id(self) -> str:
        """Get the Genesys AudioHook session ID generated by the serializer."""
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

    @property
    def output_variables(self) -> Optional[Dict[str, Any]]:
        """Get custom output variables to send back to Genesys."""
        return self._output_variables

    def set_output_variables(self, variables: Dict[str, Any]) -> None:
        """Set custom output variables to send back to Genesys on close.

        These variables will be included in the 'closed' response when Genesys
        closes the connection, making them available in the Architect flow.

        Args:
            variables: Dictionary of custom variables to send to Genesys.

        Example:
            ```python
            # During the conversation, collect data and set it
            serializer.set_output_variables({
                "intent": "billing_inquiry",
                "customer_verified": True,
                "summary": "Customer asked about their bill",
                "transfer_to": "billing_queue"
            })
            ```
        """
        self._output_variables = variables
        logger.debug(f"Output variables set: {variables}")

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate
        logger.debug(f"GenesysAudioHookSerializer setup with sample_rate={self._sample_rate}")

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
    ) -> Dict[str, Any]:
        """Create an 'opened' response message for the client.

        This should be sent in response to an 'open' message from Genesys.

        Args:
            start_paused: Whether to start the session paused.
            supported_languages: List of supported language codes.
            selected_language: The selected language code.

        Returns:
            Dictionary of the opened response message.
        """
        # Build channels list based on configuration
        channels: list[str] = []

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

        return msg

    def create_closed_response(
        self,
        output_variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a 'closed' response message.

        This should be sent in response to a 'close' message from Genesys.

        Args:
            output_variables: Optional custom variables to pass back to Genesys.
                These will be available in the Architect flow after the AudioHook
                action completes.

        Returns:
            Dictionary of the closed response message.

        Example:
            ```python
            # Pass custom data back to Genesys
            serializer.create_closed_response(
                output_variables={
                    "intent": "billing_inquiry",
                    "customer_verified": True,
                    "summary": "Customer asked about their bill"
                }
            )
            ```
        """
        parameters: Optional[Dict[str, Any]] = None

        if output_variables:
            parameters = {"outputVariables": output_variables}

        msg = self._create_message(
            AudioHookMessageType.CLOSED,
            parameters=parameters,
        )

        self._is_open = False
        logger.debug(f"AudioHook session closed: {self._session_id}")

        return msg

    def create_pong_response(self) -> Dict[str, Any]:
        """Create a 'pong' response message.

        This should be sent in response to a 'ping' message from Genesys.

        Returns:
            Dictionary of the pong response message.
        """
        msg = self._create_message(AudioHookMessageType.PONG)
        return msg

    def create_resumed_response(self) -> Dict[str, Any]:
        """Create a 'resumed' response message.

        This should be sent in response to a 'pause' message when ready to resume.

        Returns:
            Dictionary of the resumed response message.
        """
        msg = self._create_message(AudioHookMessageType.RESUMED)

        self._is_paused = False
        logger.debug(f"AudioHook session resumed: {self._session_id}")

        return msg

    def create_barge_in_event(self) -> Dict[str, Any]:
        """Create a barge-in event message.

        This notifies Genesys Cloud that the user has interrupted the bot's
        audio output. Genesys will stop any queued audio playback.

        Returns:
            Dictionary of the barge-in event message.
        """
        msg = self._create_message(
            AudioHookMessageType.EVENT,
            parameters={"entities": [{"type": "barge_in", "data": {}}]},
        )

        logger.debug("🔇 Barge-in event sent to Genesys")

        return msg

    def create_disconnect_message(
        self,
        reason: str = "completed",
        action: str = "transfer",
        output_variables: Optional[Dict[str, Any]] = None,
        info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a 'disconnect' message to initiate session termination.

        Args:
            reason: Disconnect reason (e.g., "completed", "error").
            action: Action to take ("transfer" to agent, "finished" if completed).
            output_variables: Custom output variables to pass back to Genesys.
            info: Optional additional information.

        Returns:
            Dictionary of the disconnect message.
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
        return msg

    def create_error_message(
        self,
        code: int,
        message: str,
        retryable: bool = False,
    ) -> Dict[str, Any]:
        """Create an 'error' message.

        Args:
            code: Error code.
            message: Error message.
            retryable: Whether the operation can be retried.

        Returns:
            Dictionary of the error message.
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
        return msg

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Genesys AudioHook format.

        Handles conversion of various frame types to AudioHook messages:
        - AudioRawFrame -> Binary PCMU audio data (resampled to 8kHz)
        - EndFrame/CancelFrame -> Disconnect message (JSON)
        - InterruptionFrame -> Barge-in event (JSON)
        - OutputTransportMessageFrame -> Pass-through JSON

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string (JSON) or bytes (audio), or None if
            the frame type is not handled or session is not open.
        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            return json.dumps(
                self.create_disconnect_message(
                    output_variables=self.output_variables, reason="completed"
                )
            )

        elif isinstance(frame, AudioRawFrame):
            if not self._is_open or self._is_paused:
                return None

            data = frame.audio

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

        elif isinstance(frame, InterruptionFrame):
            return json.dumps(self.create_barge_in_event())

        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            # Only pass through AudioHook protocol messages (those with "version" field)
            # Filter out RTVI and other non-AudioHook messages
            if isinstance(frame.message, dict) and "version" in frame.message:
                return json.dumps(frame.message)
            else:
                # Not an AudioHook message, ignore
                return None

        # Ignore other frames - we don't need to process them here
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Genesys AudioHook data to Pipecat frames.

        Handles:
        - Binary data -> InputAudioRawFrame (converted from PCMU to PCM)
        - JSON 'open' -> OutputTransportMessageUrgentFrame with 'opened' response
        - JSON 'close' -> OutputTransportMessageUrgentFrame with 'closed' response
        - JSON 'ping' -> OutputTransportMessageUrgentFrame with 'pong' response
        - JSON 'pause' -> Sets is_paused=True, returns None
        - JSON 'dtmf' -> InputDTMFFrame
        - JSON 'update' -> Updates participant info, returns None
        - JSON 'error' -> Logs error, returns None

        Protocol responses (opened, closed, pong) are returned as urgent frames
        to be sent immediately through the transport.

        Args:
            data: The raw WebSocket data from Genesys (binary audio or JSON text).

        Returns:
            A Pipecat frame to process, or None if handled internally.
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
        # Stereo audio comes interleaved: [L0, R0, L1, R1, ...]
        if self._params.channel == AudioHookChannel.BOTH and len(data) > 0:
            # For PCMU, each sample is 1 byte
            # Extract only bytes at even positions (left channel = external)
            audio_data = bytes(data[i] for i in range(0, len(data), 2))
            logger.debug(
                f"🔊 Stereo audio: {original_len} bytes → {len(audio_data)} bytes (external channel)"
            )

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
            logger.debug("Playback started (from Genesys)")
            return None

        elif msg_type == "playback_completed":
            logger.debug("Playback completed (from Genesys)")
            return None
        else:
            logger.warning(f"Unknown AudioHook message type: {msg_type}")
            return None

    async def _handle_open(self, message: Dict[str, Any]) -> Frame | None:
        """Handle an 'open' message from Genesys.

        This initializes the session with metadata from Genesys Cloud and
        automatically responds with an 'opened' message using the configured
        InputParams (supported_languages, selected_language, start_paused).

        Extracts and stores:
        - session_id: The AudioHook session identifier
        - conversation_id: The Genesys conversation ID
        - participant: Caller info (ani, dnis, etc.)
        - input_variables: Custom variables from Architect flow
        - media_info: Audio configuration from Genesys

        Args:
            message: The open message from Genesys.

        Returns:
            OutputTransportMessageUrgentFrame with the 'opened' response.
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
            logger.debug(
                f"📡 Genesys audio config: format={audio_media.get('format')}, channels={channels}, rate={audio_media.get('rate')}"
            )
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

        await self._call_event_handler("on_open", message)

        return OutputTransportMessageUrgentFrame(
            message=self.create_opened_response(
                start_paused=self._params.start_paused,
                supported_languages=self._params.supported_languages,
                selected_language=self._params.selected_language,
            )
        )

    async def _handle_close(self, message: Dict[str, Any]) -> Frame | None:
        """Handle a 'close' message from Genesys.

        Automatically responds with a 'closed' message. If output_variables
        were set via set_output_variables(), they will be included in the
        response and made available in the Architect flow.

        Args:
            message: The close message from Genesys.

        Returns:
            OutputTransportMessageUrgentFrame with the closed response
            (includes outputVariables if set).
        """
        params = message.get("parameters", {})
        reason = params.get("reason", "unknown")

        logger.info(f"🔴 Genesys closed the connection: {reason}")

        self._is_open = False

        logger.info(f"Sending closed response to Genesys...")

        await self._call_event_handler("on_close", message)

        # Return as urgent frame to be sent through pipeline immediately
        # Include any output variables that were set during the session
        return OutputTransportMessageUrgentFrame(
            message=self.create_closed_response(output_variables=self._output_variables)
        )

    async def _handle_ping(self, message: Dict[str, Any]) -> Frame | None:
        """Handle a 'ping' message from Genesys.

        Automatically responds with a 'pong' message to maintain the connection.

        Args:
            message: The ping message from Genesys.

        Returns:
            OutputTransportMessageUrgentFrame with pong response.
        """
        logger.info(f"Sending pong response to Genesys...")

        await self._call_event_handler("on_ping", message)

        # Return as urgent frame to be sent through pipeline immediately
        return OutputTransportMessageUrgentFrame(message=self.create_pong_response())

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

        await self._call_event_handler("on_pause", message)

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

        await self._call_event_handler("on_update", message)

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

        await self._call_event_handler("on_error", message)

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

        await self._call_event_handler("on_dtmf", message)

        try:
            return InputDTMFFrame(KeypadEntry(digit))
        except ValueError:
            # Invalid digit
            logger.warning(f"Invalid DTMF digit: {digit}")
            return None
