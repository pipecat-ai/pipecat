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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional
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
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
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
    """Map AssemblyAI language codes to Pipecat Language enum.

    AssemblyAI returns simple language codes like "es", "fr", etc.
    This function maps them to the corresponding Language enum values.

    Args:
        language_code: AssemblyAI language code (e.g., "es", "fr", "de")

    Returns:
        Corresponding Language enum value, defaulting to Language.EN if not found.
    """
    try:
        # Try to match the language code directly
        return Language(language_code.lower())
    except ValueError:
        logger.warning(
            f"Unknown language code from AssemblyAI: {language_code}, defaulting to English"
        )
        return Language.EN


@dataclass
class AssemblyAISTTSettings(STTSettings):
    """Settings for AssemblyAISTTService.

    Parameters:
        formatted_finals: Whether to enable transcript formatting.
        word_finalization_max_wait_time: Maximum time to wait for word
            finalization in milliseconds.
        end_of_turn_confidence_threshold: Confidence threshold for
            end-of-turn detection.
        min_turn_silence: Minimum silence duration when confident about
            end-of-turn.
        max_turn_silence: Maximum silence duration before forcing
            end-of-turn.
        keyterms_prompt: List of key terms to guide transcription.
        prompt: Optional text prompt to guide the transcription. Only
            used when model is "u3-rt-pro".
        language_detection: Enable automatic language detection.
        format_turns: Whether to format transcript turns.
        speaker_labels: Enable speaker diarization.
        vad_threshold: VAD confidence threshold (0.0–1.0) for classifying
            audio frames as silence. Only applicable to u3-rt-pro.
    """

    formatted_finals: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    word_finalization_max_wait_time: int | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    end_of_turn_confidence_threshold: float | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    min_turn_silence: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_turn_silence: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keyterms_prompt: List[str] | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_detection: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    format_turns: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speaker_labels: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_threshold: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AssemblyAISTTService(WebsocketSTTService):
    """AssemblyAI real-time speech-to-text service.

    Provides real-time speech transcription using AssemblyAI's WebSocket API.
    Supports both interim and final transcriptions with configurable parameters
    for audio processing and connection management.
    """

    Settings = AssemblyAISTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        language: Optional[Language] = None,
        api_endpoint_base_url: str = "wss://streaming.assemblyai.com/v3/ws",
        sample_rate: int = 16000,
        encoding: str = "pcm_s16le",
        connection_params: Optional[AssemblyAIConnectionParams] = None,
        vad_force_turn_endpoint: bool = True,
        should_interrupt: bool = True,
        speaker_format: Optional[str] = None,
        settings: Optional[Settings] = None,
        ttfs_p99_latency: Optional[float] = ASSEMBLYAI_TTFS_P99,
        **kwargs,
    ):
        """Initialize the AssemblyAI STT service.

        Args:
            api_key: AssemblyAI API key for authentication.
            language: Language code for transcription. Defaults to English (Language.EN).

                .. deprecated:: 0.0.105
                    Use ``settings=AssemblyAISTTService.Settings(language=...)`` instead.

            api_endpoint_base_url: WebSocket endpoint URL. Defaults to AssemblyAI's streaming endpoint.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            encoding: Audio encoding format. Defaults to "pcm_s16le".
            connection_params: Connection configuration parameters.

                .. deprecated:: 0.0.105
                    Use ``settings=AssemblyAISTTService.Settings(...)`` instead.

            vad_force_turn_endpoint: Controls turn detection mode.
                When True (Pipecat mode, default): Forces AssemblyAI to return finals ASAP
                so Pipecat's turn detection (e.g., Smart Turn) decides when the user is done.
                - min_turn_silence defaults to 100ms (user can override)
                - max_turn_silence is ALWAYS set equal to min_turn_silence
                - VAD stop sends ForceEndpoint as ceiling
                - No UserStarted/StoppedSpeakingFrame emitted from STT
                When False (AssemblyAI turn detection mode, u3-rt-pro only): AssemblyAI's model
                controls turn endings using built-in turn detection.
                - Uses AssemblyAI API defaults for all parameters (unless user explicitly sets them)
                - Emits UserStarted/StoppedSpeakingFrame from STT
                - No ForceEndpoint on VAD stop
            should_interrupt: Whether to interrupt the bot when the user starts speaking
                in AssemblyAI turn detection mode (vad_force_turn_endpoint=False). Only applies
                when using AssemblyAI's built-in turn detection. Defaults to True.
            speaker_format: Optional format string for speaker labels when diarization is enabled.
                Use {speaker} for speaker label and {text} for transcript text.
                Example: "<{speaker}>{text}</{speaker}>" or "{speaker}: {text}"
                If None, transcript text is not modified. Defaults to None.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to parent STTService class.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="u3-rt-pro",
            language=Language.EN,
            formatted_finals=True,
            word_finalization_max_wait_time=None,
            end_of_turn_confidence_threshold=None,
            min_turn_silence=None,
            max_turn_silence=None,
            keyterms_prompt=None,
            prompt=None,
            language_detection=None,
            format_turns=True,
            speaker_labels=None,
            vad_threshold=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if language is not None:
            self._warn_init_param_moved_to_settings("language", "language")
            default_settings.language = language

        # 3. Apply connection_params overrides (deprecated) — only if settings not provided
        if connection_params is not None:
            self._warn_init_param_moved_to_settings("connection_params")
            if not settings:
                sample_rate = connection_params.sample_rate
                encoding = connection_params.encoding
                default_settings.model = connection_params.speech_model
                default_settings.formatted_finals = connection_params.formatted_finals
                default_settings.word_finalization_max_wait_time = (
                    connection_params.word_finalization_max_wait_time
                )
                default_settings.end_of_turn_confidence_threshold = (
                    connection_params.end_of_turn_confidence_threshold
                )
                default_settings.min_turn_silence = connection_params.min_turn_silence
                default_settings.max_turn_silence = connection_params.max_turn_silence
                default_settings.keyterms_prompt = connection_params.keyterms_prompt
                default_settings.prompt = connection_params.prompt
                default_settings.language_detection = connection_params.language_detection
                default_settings.format_turns = connection_params.format_turns
                default_settings.speaker_labels = connection_params.speaker_labels
                default_settings.vad_threshold = connection_params.vad_threshold

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # 5. Validate final settings
        is_u3_pro = default_settings.model == "u3-rt-pro"
        if not vad_force_turn_endpoint and not is_u3_pro:
            raise ValueError(
                f"AssemblyAI turn detection mode (vad_force_turn_endpoint=False) requires "
                f"u3-rt-pro for SpeechStarted support. Either set "
                f"vad_force_turn_endpoint=True for {default_settings.model}, "
                f"or use model='u3-rt-pro'."
            )

        if default_settings.prompt is not None and default_settings.keyterms_prompt is not None:
            raise ValueError(
                "The prompt and keyterms_prompt parameters cannot be used in the same request. "
                "Please choose either one or the other based on your use case. When you use "
                "keyterms_prompt, your boosted words are appended to the default prompt automatically. "
                "Or to boost within prompt: <prompt> + Make sure to boost the words <keyterms> "
                "in the audio. "
                "For more info go to: https://www.assemblyai.com/docs/streaming/universal-3-pro"
            )

        if default_settings.prompt is not None:
            logger.warning(
                "Custom prompt detected. Prompting is a beta feature. We recommend testing "
                "with no prompt first, as this will use our optimized default prompt for "
                "voice agents. Bad prompts may lead to bad results. If you'd like to create "
                "your own prompt, check out our prompting guide at: "
                "https://www.assemblyai.com/docs/streaming/prompting"
            )

        # 6. Configure pipecat turn mode (mutates default_settings)
        if vad_force_turn_endpoint:
            self._configure_pipecat_turn_mode(default_settings, is_u3_pro)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._api_endpoint_base_url = api_endpoint_base_url
        self._vad_force_turn_endpoint = vad_force_turn_endpoint
        self._should_interrupt = should_interrupt
        self._speaker_format = speaker_format

        # Init-only audio config (not runtime-updatable)
        self._encoding = encoding

        self._termination_event = asyncio.Event()
        self._received_termination = False
        self._connected = False

        self._receive_task = None

        self._audio_buffer = bytearray()
        self._chunk_size_ms = 50
        self._chunk_size_bytes = 0

        self._user_speaking = False

    def _configure_pipecat_turn_mode(self, settings: Settings, is_u3_pro: bool):
        """Configure settings for Pipecat turn detection mode.

        When vad_force_turn_endpoint is enabled, force AssemblyAI to return
        finals as fast as possible so Pipecat's smart turn analyzer can decide
        when the user is done speaking. VAD stop is the absolute ceiling.

        u3-rt-pro:
        - min_turn_silence defaults to 100ms (user can override)
        - max_turn_silence is ALWAYS set equal to min_turn_silence
          to avoid double turn detection (AssemblyAI + Pipecat both analyzing)
        - If user sets max_turn_silence, it's ignored with a warning
        - end_of_turn_confidence_threshold: not set (API default)

        universal-streaming-*:
        - end_of_turn_confidence_threshold=0.0 (disable semantic turn detection)
        - min_turn_silence=160
        - max_turn_silence: not set (API default)

        Args:
            settings: The settings to configure in place.
            is_u3_pro: Whether using u3-rt-pro model.
        """
        if is_u3_pro:
            # u3-rt-pro: Synchronize max_turn_silence with min_turn_silence
            min_silence = settings.min_turn_silence
            if min_silence is None:
                min_silence = 100

            # Warn if user set max_turn_silence (will be overridden)
            if settings.max_turn_silence is not None:
                logger.warning(
                    f"Your max_turn_silence value ({settings.max_turn_silence}ms) will be "
                    f"OVERRIDDEN in Pipecat mode (vad_force_turn_endpoint=True). It will be set to "
                    f"{min_silence}ms (matching min_turn_silence) and SENT to "
                    f"AssemblyAI to avoid double turn detection. To use your max_turn_silence as-is, "
                    f"switch to AssemblyAI turn detection mode (vad_force_turn_endpoint=False)."
                )

            settings.min_turn_silence = min_silence
            settings.max_turn_silence = min_silence
        else:
            # universal-streaming: Different configuration (works differently)
            settings.end_of_turn_confidence_threshold = 1.0
            settings.min_turn_silence = 160

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta and reconnect to apply changes.

        Args:
            delta: A settings delta with updated values.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        # Reconnect to apply updated settings (they become WS query params)
        await self._disconnect()
        await self._connect()

        return changed

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

        Args:
            frame: Frame to process.
            direction: Direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            pass
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if (
                self._vad_force_turn_endpoint
                and self._websocket
                and self._websocket.state is State.OPEN
            ):
                self.request_finalize()
                await self._websocket.send(json.dumps({"type": "ForceEndpoint"}))
            await self.start_processing_metrics()

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    def _build_ws_url(self) -> str:
        """Build WebSocket URL with query parameters using urllib.parse.urlencode."""
        s = self._settings
        params: dict[str, Any] = {}

        # Init-only audio config
        params["sample_rate"] = self.sample_rate
        params["encoding"] = self._encoding

        # Map model → speech_model (AssemblyAI API naming)
        if s.model is not None:
            params["speech_model"] = s.model

        # Settings fields (skip None values)
        optional_fields = {
            "formatted_finals": s.formatted_finals,
            "word_finalization_max_wait_time": s.word_finalization_max_wait_time,
            "end_of_turn_confidence_threshold": s.end_of_turn_confidence_threshold,
            "min_turn_silence": s.min_turn_silence,
            "max_turn_silence": s.max_turn_silence,
            "prompt": s.prompt,
            "language_detection": s.language_detection,
            "format_turns": s.format_turns,
            "speaker_labels": s.speaker_labels,
            "vad_threshold": s.vad_threshold,
        }

        for k, v in optional_fields.items():
            if v is not None:
                if isinstance(v, bool):
                    params[k] = str(v).lower()
                else:
                    params[k] = v

        # Special handling for keyterms_prompt (needs JSON encoding)
        if s.keyterms_prompt is not None:
            params["keyterms_prompt"] = json.dumps(s.keyterms_prompt)

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
                # Log raw JSON for Turn messages to debug speaker_label
                if data.get("type") == "Turn":
                    logger.trace(f"{self} RAW JSON from AssemblyAI: {json.dumps(data, indent=2)}")
                await self._handle_message(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")

    def _parse_message(self, message: Dict[str, Any]) -> BaseMessage:
        """Parse a raw message into the appropriate message type."""
        msg_type = message.get("type")

        if msg_type == "Begin":
            return BeginMessage.model_validate(message)
        elif msg_type == "Turn":
            return TurnMessage.model_validate(message)
        elif msg_type == "SpeechStarted":
            return SpeechStartedMessage.model_validate(message)
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
            elif isinstance(parsed_message, TurnMessage):
                await self._handle_transcription(parsed_message)
            elif isinstance(parsed_message, SpeechStartedMessage):
                await self._handle_speech_started(parsed_message)
            elif isinstance(parsed_message, TerminationMessage):
                await self._handle_termination(parsed_message)
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    async def _handle_speech_started(self, message: SpeechStartedMessage):
        """Handle SpeechStarted event — fast barge-in for AssemblyAI turn detection.

        Broadcasts UserStartedSpeakingFrame to signal the start of user
        speech, then pushes an interruption to cancel any bot audio.
        SpeechStarted fires before any transcript arrives, so the turn
        is cleanly started before any transcription frames are pushed.

        Only applies when using AssemblyAI's built-in turn detection. When using
        Pipecat turn detection, VAD + smart turn analyzer handle interruptions.
        """
        if self._vad_force_turn_endpoint:
            return  # Pipecat mode: handled by aggregator

        await self.start_processing_metrics()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.broadcast_interruption()
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
        """Handle transcription results with two turn detection modes.

        Pipecat turn detection (vad_force_turn_endpoint=True):
            - No UserStarted/StoppedSpeakingFrame from STT
            - end_of_turn → TranscriptionFrame (finalized set by base class
              if this is a ForceEndpoint response)
            - else → InterimTranscriptionFrame

        AssemblyAI turn detection (vad_force_turn_endpoint=False):
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

        # Handle speaker diarization
        speaker_id = self._user_id
        transcript_text = message.transcript

        if message.speaker:
            speaker_id = message.speaker
            # Format transcript with speaker labels if format string provided
            if self._speaker_format:
                transcript_text = self._speaker_format.format(
                    speaker=message.speaker, text=message.transcript
                )

        # Determine if this is a final turn from AssemblyAI
        is_final_turn = message.end_of_turn and (
            not self._settings.format_turns or message.turn_is_formatted
        )

        if self._vad_force_turn_endpoint:
            # --- Pipecat turn detection mode ---
            # No UserStarted/StoppedSpeakingFrame — VAD + smart turn analyzer handle this
            if is_final_turn:
                finalize_confirmed = bool(message.turn_is_formatted)
                if finalize_confirmed:
                    self.confirm_finalize()
                logger.debug(f'{self} Transcript: "{transcript_text}"')
                await self.push_frame(
                    TranscriptionFrame(
                        transcript_text,
                        speaker_id,
                        time_now_iso8601(),
                        language,
                        message,
                    )
                )
                await self._trace_transcription(transcript_text, True, language)
                await self.stop_processing_metrics()
            else:
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript_text,
                        speaker_id,
                        time_now_iso8601(),
                        language,
                        message,
                    )
                )
        else:
            # --- AssemblyAI turn detection mode ---
            # SpeechStarted always arrives before transcripts with u3-rt-pro,
            # so UserStartedSpeakingFrame is guaranteed to be broadcast first.
            if is_final_turn:
                # AssemblyAI controls finalization, just mark as finalized
                await self.push_frame(
                    TranscriptionFrame(
                        transcript_text,
                        speaker_id,
                        time_now_iso8601(),
                        language,
                        message,
                        finalized=True,
                    )
                )
                await self._trace_transcription(transcript_text, True, language)
                await self.stop_processing_metrics()
                # AAI is authoritative — emit UserStoppedSpeakingFrame immediately.
                # broadcast_frame pushes downstream (same queue as TranscriptionFrame
                # above, so ordering is preserved) and upstream.
                await self.broadcast_frame(UserStoppedSpeakingFrame)
                self._user_speaking = False
            else:
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript_text,
                        speaker_id,
                        time_now_iso8601(),
                        language,
                        message,
                    )
                )
