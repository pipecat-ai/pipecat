#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AssemblyAI speech-to-text service implementation.

This module provides integration with AssemblyAI's real-time speech-to-text
WebSocket API for streaming audio transcription.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional
from urllib.parse import urlencode

from loguru import logger

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_latency import ASSEMBLYAI_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

from .models import (
    AssemblyAIConnectionParams,
    BaseMessage,
    BeginMessage,
    SpeechStartedMessage,
    TerminationMessage,
    TurnMessage,
)

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use AssemblyAI, you need to `pip install "pipecat-ai[assemblyai]"`.')
    raise Exception(f"Missing module: {e}")


def map_language_from_assemblyai(language_code: str) -> Language:
    """Map AssemblyAI language code to pipecat Language enum.

    AssemblyAI streaming supports 6 languages: English, Spanish, French,
    German, Italian, and Portuguese.

    Args:
        language_code: The AssemblyAI language code (e.g., "es", "fr").

    Returns:
        The corresponding pipecat Language enum value.
    """
    LANGUAGE_MAP = {
        "de": Language.DE,
        "en": Language.EN,
        "es": Language.ES,
        "fr": Language.FR,
        "it": Language.IT,
        "pt": Language.PT,
    }

    return LANGUAGE_MAP.get(language_code.lower(), Language.EN)


class AssemblyAISTTService(WebsocketSTTService):
    """AssemblyAI real-time speech-to-text service.

    Provides real-time speech transcription using AssemblyAI's WebSocket API.
    Supports both interim and final transcriptions with configurable parameters
    for audio processing and connection management.
    """

    def __init__(
        self,
        *,
        api_key: str,
        language: Language = Language.EN,
        api_endpoint_base_url: str = "wss://streaming.assemblyai.com/v3/ws",
        connection_params: AssemblyAIConnectionParams = AssemblyAIConnectionParams(),
        vad_force_turn_endpoint: bool = True,
        should_interrupt: bool = True,
        ttfs_p99_latency: Optional[float] = ASSEMBLYAI_TTFS_P99,
        **kwargs,
    ):
        """Initialize the AssemblyAI STT service.

        Args:
            api_key: AssemblyAI API key for authentication.
            language: Language for transcription metadata. When using universal-streaming-multilingual,
                AssemblyAI auto-detects the language and transcription frames will use the detected
                language (with confidence >= 0.7), falling back to English for low confidence or
                missing detections. Supported languages: EN, ES, FR, DE, IT, PT. Defaults to
                English (Language.EN).
            api_endpoint_base_url: WebSocket endpoint URL. Defaults to AssemblyAI's streaming endpoint.
            connection_params: Connection configuration parameters. Defaults to AssemblyAIConnectionParams().
            vad_force_turn_endpoint: Controls turn detection mode.
                When True (Pipecat mode): Forces AssemblyAI to return finals ASAP
                (confidence=0.0, fast silence params) so Pipecat's smart turn analyzer
                decides when the user is done. VAD stop sends ForceEndpoint as ceiling.
                No UserStarted/StoppedSpeakingFrame emitted from STT.
                When False (STT mode, default): Respects user params for confidence and
                silence. AssemblyAI's end_of_turn controls turn endings. Emits
                UserStarted/StoppedSpeakingFrame from STT. No ForceEndpoint on VAD stop.
            should_interrupt: Whether to interrupt the bot when the user starts speaking
                in STT mode (vad_force_turn_endpoint=False). Only applies to STT mode.
                Defaults to True.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to parent STTService class.
        """
        # STT turn detection (vad_force_turn_endpoint=False) requires the
        # SpeechStarted event for reliable barge-in. Only u3-rt-pro supports
        # this. Other models must use Pipecat turn detection.
        is_u3_pro = connection_params.speech_model == "u3-rt-pro"
        if not vad_force_turn_endpoint and not is_u3_pro:
            raise ValueError(
                f"STT turn detection (vad_force_turn_endpoint=False) requires "
                f"u3-rt-pro for SpeechStarted support. Either set "
                f"vad_force_turn_endpoint=True for {connection_params.speech_model}, "
                f"or use speech_model='u3-rt-pro'."
            )

        # When vad_force_turn_endpoint is enabled, configure connection params
        # for Pipecat turn detection mode (fast finals for smart turn analyzer)
        if vad_force_turn_endpoint:
            connection_params = self._configure_pipecat_turn_mode(connection_params)

        super().__init__(
            sample_rate=connection_params.sample_rate, ttfs_p99_latency=ttfs_p99_latency, **kwargs
        )

        self._api_key = api_key
        self._language = language
        self._api_endpoint_base_url = api_endpoint_base_url
        self._connection_params = connection_params
        self._vad_force_turn_endpoint = vad_force_turn_endpoint
        self._should_interrupt = should_interrupt

        self._termination_event = asyncio.Event()
        self._received_termination = False
        self._connected = False

        self._receive_task = None

        self._audio_buffer = bytearray()
        self._chunk_size_ms = 50
        self._chunk_size_bytes = 0

        self._user_speaking = False
        self._vad_speaking = False

    def _configure_pipecat_turn_mode(
        self, connection_params: AssemblyAIConnectionParams
    ) -> AssemblyAIConnectionParams:
        """Configure connection params for Pipecat turn detection mode.

        When vad_force_turn_endpoint is enabled, force AssemblyAI to return
        finals as fast as possible so Pipecat's smart turn analyzer can decide
        when the user is done speaking. VAD stop is the absolute ceiling.

        u3-rt-pro:
        - min_end_of_turn_silence_when_confident=100
        - max_turn_silence=100
        - end_of_turn_confidence_threshold: not set (API default)

        universal-streaming-*:
        - end_of_turn_confidence_threshold=0.0
        - min_end_of_turn_silence_when_confident=160
        - max_turn_silence: not set (API default)

        Args:
            connection_params: The user-provided connection parameters.

        Returns:
            Updated connection parameters configured for Pipecat turn mode.
        """
        is_u3_pro = connection_params.speech_model == "u3-rt-pro"

        if is_u3_pro:
            updates = {
                "min_end_of_turn_silence_when_confident": 100,
                "max_turn_silence": 100,
            }
        else:
            updates = {
                "end_of_turn_confidence_threshold": 0.0,
                "min_end_of_turn_silence_when_confident": 160,
            }

        return connection_params.model_copy(update=updates)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the speech-to-text service.

        Args:
            frame: Start frame to begin processing.
        """
        await super().start(frame)
        self._chunk_size_bytes = int(self._chunk_size_ms * self.sample_rate * 2 / 1000)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the speech-to-text service.

        Args:
            frame: End frame to stop processing.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech-to-text service.

        Args:
            frame: Cancel frame to abort processing.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text conversion.

        Args:
            audio: Raw audio bytes to process.

        Yields:
            None (processing handled via WebSocket messages).
        """
        self._audio_buffer.extend(audio)

        if self._websocket and self._websocket.state is State.OPEN:
            while len(self._audio_buffer) >= self._chunk_size_bytes:
                chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
                self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
                await self._websocket.send(chunk)

        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for VAD and metrics handling.

        Mode 1 (vad_force_turn_endpoint=True): VAD stop sends ForceEndpoint +
        request_finalize() so the base class marks the response as finalized.
        Mode 2 (vad_force_turn_endpoint=False): No ForceEndpoint on VAD stop.
        AssemblyAI's max_turn_silence (synced to VAD stop_secs) is the ceiling.

        Args:
            frame: Frame to process.
            direction: Direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_speaking = True
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_speaking = False
            # Mode 1 only: ForceEndpoint on VAD stop as absolute ceiling
            if (
                self._vad_force_turn_endpoint
                and self._websocket
                and self._websocket.state is State.OPEN
            ):
                self.request_finalize()
                await self._websocket.send(json.dumps({"type": "ForceEndpoint"}))

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    def _build_ws_url(self) -> str:
        """Build WebSocket URL with query parameters using urllib.parse.urlencode."""
        params = {}
        for k, v in self._connection_params.model_dump().items():
            if v is not None:
                # Only include prompt parameter if speech_model is u3-rt-pro
                if k == "prompt" and self._connection_params.speech_model != "u3-rt-pro":
                    continue
                # Only include language_detection for models that support it
                if k == "language_detection" and self._connection_params.speech_model == "universal-streaming-english":
                    continue
                if k == "keyterms_prompt":
                    params[k] = json.dumps(v)
                elif isinstance(v, bool):
                    params[k] = str(v).lower()
                else:
                    params[k] = v

        if params:
            query_string = urlencode(params)
            return f"{self._api_endpoint_base_url}?{query_string}"
        return self._api_endpoint_base_url

    async def _connect(self):
        """Connect to the AssemblyAI service.

        Establishes websocket connection and starts receive task.
        """
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the AssemblyAI service.

        Sends termination message, waits for acknowledgment, and cleans up.
        """
        await super()._disconnect()

        if not self._connected or not self._websocket:
            return

        try:
            self._termination_event.clear()
            self._received_termination = False

            if self._websocket.state is State.OPEN:
                # Send any remaining audio
                if len(self._audio_buffer) > 0:
                    await self._websocket.send(bytes(self._audio_buffer))
                    self._audio_buffer.clear()

                # Send termination message and wait for acknowledgment
                try:
                    await self._websocket.send(json.dumps({"type": "Terminate"}))

                    try:
                        await asyncio.wait_for(self._termination_event.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning("Timed out waiting for termination message from server")

                except Exception as e:
                    await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            # Clean up tasks and connection
            if self._receive_task:
                await self.cancel_task(self._receive_task)
                self._receive_task = None

            await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the websocket connection to AssemblyAI."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to AssemblyAI WebSocket")

            ws_url = self._build_ws_url()
            headers = {
                "Authorization": self._api_key,
                "User-Agent": f"AssemblyAI/1.0 (integration=Pipecat/{pipecat_version()})",
            }
            self._websocket = await websocket_connect(
                ws_url,
                additional_headers=headers,
            )
            self._connected = True
            await self._call_event_handler("on_connected")
            logger.debug(f"{self} Connected to AssemblyAI WebSocket")
        except Exception as e:
            self._connected = False
            await self.push_error(error_msg=f"Unable to connect to AssemblyAI: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close the websocket connection to AssemblyAI."""
        try:
            if self._websocket:
                logger.debug("Disconnecting from AssemblyAI WebSocket")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            self._connected = False
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The WebSocket connection.

        Raises:
            Exception: If WebSocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process websocket messages.

        Continuously processes messages from the websocket connection.
        """
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")

    def _parse_message(self, message: Dict[str, Any]) -> BaseMessage:
        """Parse a raw message into the appropriate message type."""
        msg_type = message.get("type")

        if msg_type == "Begin":
            return BeginMessage.model_validate(message)
        elif msg_type == "SpeechStarted":
            return SpeechStartedMessage.model_validate(message)
        elif msg_type == "Turn":
            return TurnMessage.model_validate(message)
        elif msg_type == "Termination":
            return TerminationMessage.model_validate(message)
        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle AssemblyAI WebSocket messages."""
        try:
            parsed_message = self._parse_message(message)

            if isinstance(parsed_message, BeginMessage):
                logger.debug(
                    f"Session Begin: {parsed_message.id} (expires at {parsed_message.expires_at})"
                )
            elif isinstance(parsed_message, SpeechStartedMessage):
                await self._handle_speech_started(parsed_message)
            elif isinstance(parsed_message, TurnMessage):
                await self._handle_transcription(parsed_message)
            elif isinstance(parsed_message, TerminationMessage):
                await self._handle_termination(parsed_message)
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    async def _handle_speech_started(self, message: SpeechStartedMessage):
        """Handle SpeechStarted event — fast barge-in for Mode 2.

        Broadcasts UserStartedSpeakingFrame to signal the start of user
        speech, then pushes an interruption to cancel any bot audio.
        SpeechStarted fires before any transcript arrives, so the turn
        is cleanly started before any transcription frames are pushed.

        Only applies to Mode 2 (STT turn detection). In Mode 1, VAD +
        smart turn analyzer handle interruptions via the aggregator.
        """
        if self._vad_force_turn_endpoint:
            return  # Mode 1: handled by aggregator

        await self.start_processing_metrics()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.push_interruption_task_frame_and_wait()
        self._user_speaking = True

    async def _handle_termination(self, message: TerminationMessage):
        """Handle termination message."""
        self._received_termination = True
        self._termination_event.set()

        logger.info(
            f"Session Terminated: Audio Duration={message.audio_duration_seconds}s, "
            f"Session Duration={message.session_duration_seconds}s"
        )
        await self.push_frame(EndFrame())

    async def _handle_transcription(self, message: TurnMessage):
        """Handle transcription results with two-mode turn detection.

        Mode 1 (vad_force_turn_endpoint=True, Pipecat turn detection):
            - No UserStarted/StoppedSpeakingFrame from STT
            - end_of_turn → TranscriptionFrame (finalized set by base class
              if this is a ForceEndpoint response)
            - else → InterimTranscriptionFrame

        Mode 2 (vad_force_turn_endpoint=False, STT turn detection):
            - UserStartedSpeakingFrame on first transcript
            - end_of_turn → TranscriptionFrame + UserStoppedSpeakingFrame
            - else → InterimTranscriptionFrame
        """
        if not message.transcript:
            return

        # Use detected language if available with sufficient confidence
        language = Language.EN
        if message.language_code and message.language_confidence:
            if message.language_confidence >= 0.7:
                language = map_language_from_assemblyai(message.language_code)
            else:
                logger.warning(
                    f"Low language detection confidence ({message.language_confidence:.2f}) "
                    f"for language '{message.language_code}', falling back to English"
                )

        # Determine if this is a final turn from AssemblyAI
        is_final_turn = message.end_of_turn and (
            not self._connection_params.format_turns or message.turn_is_formatted
        )

        if self._vad_force_turn_endpoint:
            # --- Mode 1: Pipecat turn detection ---
            # No UserStarted/StoppedSpeakingFrame — VAD + smart turn analyzer handle this
            if is_final_turn:
                finalize_confirmed = bool(message.turn_is_formatted)
                if finalize_confirmed:
                    self.confirm_finalize()
                logger.debug(f"{self} Final transcript: \"{message.transcript}\"")
                await self.push_frame(
                    TranscriptionFrame(
                        message.transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        message,
                    )
                )
                await self._trace_transcription(message.transcript, True, language)
                await self.stop_processing_metrics()
            else:
                logger.debug(f"{self} Interim transcript: \"{message.transcript}\"")
                await self.push_frame(
                    InterimTranscriptionFrame(
                        message.transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        message,
                    )
                )
        else:
            # --- Mode 2: STT turn detection ---
            # SpeechStarted handles UserStartedSpeakingFrame + interruption.
            # If SpeechStarted hasn't fired yet (shouldn't happen, but guard),
            # broadcast here as fallback.
            if not self._user_speaking:
                logger.warning(f"{self} Transcript arrived before SpeechStarted, broadcasting fallback UserStartedSpeakingFrame")
                await self.broadcast_frame(UserStartedSpeakingFrame)
                self._user_speaking = True

            if is_final_turn:
                if message.turn_is_formatted:
                    self.confirm_finalize()
                await self.push_frame(
                    TranscriptionFrame(
                        message.transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        message,
                        finalized=True,
                    )
                )
                await self._trace_transcription(message.transcript, True, language)
                await self.stop_processing_metrics()
                # AAI is authoritative — emit UserStoppedSpeakingFrame immediately.
                # broadcast_frame pushes downstream (same queue as TranscriptionFrame
                # above, so ordering is preserved) and upstream.
                await self.broadcast_frame(UserStoppedSpeakingFrame)
                self._user_speaking = False
            else:
                await self.push_frame(
                    InterimTranscriptionFrame(
                        message.transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        message,
                    )
                )
