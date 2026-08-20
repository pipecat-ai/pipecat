#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam AI Speech-to-Text service implementations.

Both services stream audio to Sarvam's WebSocket API for Indian language speech
recognition. :class:`SarvamSTTService` covers the transcription endpoint, with
Voice Activity Detection and a choice of audio formats.
:class:`SarvamRealtimeSTTService` targets the realtime endpoint, which adds
server-side endpointing and in-band configuration updates.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, fields
from typing import Any, Literal, cast
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import SARVAM_REALTIME_TTFS_P99, SARVAM_TTFS_P99
from pipecat.services.stt_service import STTService, WebsocketSTTService
from pipecat.services.websocket_service import ReportErrorCallback
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.deprecation import deprecated
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given

try:
    from sarvamai import AsyncSarvamAI
    from sarvamai.core.api_error import ApiError
    from sarvamai.core.events import EventType
    from sarvamai.core.request_options import RequestOptions
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Sarvam, you need to `uv add "pipecat-ai[sarvam]"`.')
    raise ImportError(f"Missing module: {e}") from e


def language_to_sarvam_language(language: Language) -> str:
    """Convert a Language enum to Sarvam's language code format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Sarvam language code string.
    """
    # Mapping of pipecat Language enum to Sarvam language codes
    LANGUAGE_MAP = {
        Language.BN_IN: "bn-IN",
        Language.GU_IN: "gu-IN",
        Language.HI_IN: "hi-IN",
        Language.KN_IN: "kn-IN",
        Language.ML_IN: "ml-IN",
        Language.MR_IN: "mr-IN",
        Language.TA_IN: "ta-IN",
        Language.TE_IN: "te-IN",
        Language.PA_IN: "pa-IN",
        Language.OR_IN: "od-IN",
        Language.EN_IN: "en-IN",
        Language.AS_IN: "as-IN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


SarvamMode = Literal["transcribe", "translate", "verbatim", "translit", "codemix"]


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for a Sarvam STT model.

    Parameters:
        supports_mode: Whether the model accepts mode parameter.
        supports_language: Whether the model accepts language parameter.
        default_language: Default language code (None = auto-detect).
        default_mode: Default mode (None = not applicable).
    """

    supports_mode: bool
    supports_language: bool
    default_language: str | None
    default_mode: SarvamMode | None


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "saaras:v3": ModelConfig(
        supports_mode=True,
        supports_language=True,
        default_language="unknown",
        default_mode="transcribe",
    ),
    "saaras:v4": ModelConfig(
        supports_mode=True,
        supports_language=True,
        default_language="unknown",
        default_mode="transcribe",
    ),
}


@dataclass
class SarvamSTTSettings(STTSettings):
    """Settings for SarvamSTTService.

    Parameters:
        vad_signals: Enable VAD signals in response.
        high_vad_sensitivity: Enable high VAD sensitivity.
        positive_speech_threshold: VAD probability threshold (0.0-1.0) above which
            a frame is considered speech.
        negative_speech_threshold: VAD probability threshold (0.0-1.0) below which
            a frame is considered silence.
        min_speech_frames: Minimum consecutive speech frames to start a speech segment.
        first_turn_min_speech_frames: Minimum speech frames for the first user turn.
        negative_frames_count: Number of silence frames within the window to end
            a speech segment.
        negative_frames_window: Sliding window size (in frames) for counting
            negative frames.
        start_speech_volume_threshold: Volume level (dB) below which audio is
            too quiet to be speech.
        interrupt_min_speech_frames: Minimum speech frames to register a
            barge-in/interruption.
        pre_speech_pad_frames: Number of audio frames to prepend before detected
            speech onset.
        num_initial_ignored_frames: Number of leading audio frames to skip at
            connection start.
    """

    vad_signals: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    high_vad_sensitivity: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    positive_speech_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    negative_speech_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_speech_frames: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    first_turn_min_speech_frames: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    negative_frames_count: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    negative_frames_window: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    start_speech_volume_threshold: float | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    interrupt_min_speech_frames: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pre_speech_pad_frames: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    num_initial_ignored_frames: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SarvamSTTService(STTService):
    """Sarvam speech-to-text service.

    Provides real-time speech recognition using Sarvam's WebSocket API.

    Event handlers available (in addition to STTService events):

    - on_connected(service): Connected to Sarvam WebSocket
    - on_disconnected(service): Disconnected from Sarvam WebSocket
    - on_connection_error(service, error): Connection error occurred

    Example::

        @stt.event_handler("on_connected")
        async def on_connected(service):
            ...
    """

    Settings = SarvamSTTSettings
    _settings: Settings

    @deprecated(
        "`SarvamSTTService.InputParams` is deprecated since 0.0.105 and will be removed in 2.0.0. "
        "Use `SarvamSTTService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for Sarvam STT service.

        .. deprecated:: 0.0.105
            Use ``settings=SarvamSTTService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Target language for transcription.
                - saaras:v3: Defaults to "unknown" (auto-detect supported)
                - saaras:v4: Defaults to "unknown" (auto-detect supported)
            mode: Mode of operation for models that support it. Options: transcribe,
                translate, verbatim, translit, codemix. Defaults to "transcribe".
            vad_signals: Enable VAD signals in response. Defaults to None.
            high_vad_sensitivity: Enable high VAD sensitivity. Defaults to None.
        """

        language: Language | None = None
        mode: SarvamMode | None = None
        vad_signals: bool | None = None
        high_vad_sensitivity: bool | None = None

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        mode: SarvamMode | None = None,
        sample_rate: int | None = None,
        input_audio_codec: str = "wav",
        params: InputParams | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = SARVAM_TTFS_P99,
        keepalive_timeout: float | None = None,
        keepalive_interval: float = 5.0,
        **kwargs,
    ):
        """Initialize the Sarvam STT service.

        Args:
            api_key: Sarvam API key for authentication.
            model: Sarvam model to use for transcription.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamSTTService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            mode: Mode of operation. Options: transcribe, translate, verbatim,
                translit, codemix. Only applicable to models that support it.
                Defaults to the model's default mode.
            sample_rate: Audio sample rate. Defaults to 16000 if not specified.
            input_audio_codec: Audio codec/format of the input file. Defaults to "wav".
            params: Configuration parameters for Sarvam STT service.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamSTTService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            keepalive_timeout: Seconds of no audio before sending silence to keep the
                connection alive. None disables keepalive.
            keepalive_interval: Seconds between idle checks when keepalive is enabled.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(
            model="saaras:v4",
            language=None,
            vad_signals=None,
            high_vad_sensitivity=None,
            positive_speech_threshold=None,
            negative_speech_threshold=None,
            min_speech_frames=None,
            first_turn_min_speech_frames=None,
            negative_frames_count=None,
            negative_frames_window=None,
            start_speech_volume_threshold=None,
            interrupt_min_speech_frames=None,
            pre_speech_pad_frames=None,
            num_initial_ignored_frames=None,
        )

        # --- 2. Deprecated direct-arg overrides ---
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # --- 3. Deprecated params overrides ---
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language
                if params.mode is not None:
                    mode = params.mode
                default_settings.vad_signals = params.vad_signals
                default_settings.high_vad_sensitivity = params.high_vad_sensitivity

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        # Resolve model config and validate (after all overrides)
        resolved_model = assert_given(default_settings.model)
        if resolved_model is None or resolved_model not in MODEL_CONFIGS:
            allowed = ", ".join(sorted(MODEL_CONFIGS.keys()))
            raise ValueError(f"Unsupported model '{resolved_model}'. Allowed values: {allowed}.")

        self._config = MODEL_CONFIGS[resolved_model]

        # Validate parameters against model capabilities
        if mode is not None and not self._config.supports_mode:
            raise ValueError(f"Model '{resolved_model}' does not support mode parameter.")
        if default_settings.language is not None and not self._config.supports_language:
            raise ValueError(
                f"Model '{resolved_model}' does not support language parameter (auto-detects language)."
            )

        # Resolve mode default from model config
        if mode is None:
            mode = self._config.default_mode

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=keepalive_timeout,
            keepalive_interval=keepalive_interval,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key

        # Init-only connection config (not runtime-updatable)
        self._mode = mode

        # Store connection parameters
        self._input_audio_codec = input_audio_codec

        # Initialize Sarvam SDK client
        self._sdk_headers = sdk_headers()
        # Pass Pipecat SDK headers directly at client construction time so they are
        # merged by the Sarvam SDK's client wrapper and consistently applied to
        # WebSocket handshake requests.
        self._sarvam_client = AsyncSarvamAI(api_subscription_key=api_key, headers=self._sdk_headers)
        self._websocket_context = None
        self._socket_client = None
        self._receive_task = None

        if default_settings.vad_signals:
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_speech_stopped")
            self._register_event_handler("on_utterance_end")

        logger.info(f"Sarvam STT initialized with SDK headers: {self._sdk_headers}")

    def language_to_service_language(self, language: Language) -> str:
        """Convert pipecat Language enum to Sarvam's language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The Sarvam language code string.
        """
        return language_to_sarvam_language(language)

    def _get_language_string(self) -> str | None:
        """Resolve the current language setting to a Sarvam language code string."""
        # The stored language is a Sarvam code rather than a Language, but the
        # mapping keys compare equal either way.
        language = cast(Language, assert_given(self._settings.language))
        if language:
            return language_to_sarvam_language(language)
        return self._config.default_language

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Request external turn strategies when Sarvam's VAD signals drive turns.

        With ``vad_signals`` enabled Sarvam detects speech boundaries server-side
        and this service proposes turns from them, so the user aggregator resolves
        those rather than running local VAD/smart-turn. Without it the defaults are
        left in place. Applied unless the user passed their own
        ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        if self._settings.vad_signals:
            frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames.

        Flushes on Pipecat's VAD turn end when Sarvam's built-in VAD signals
        aren't driving the turn.
        """
        await super().process_frame(frame, direction)

        # Only handle VAD frames when not using Sarvam's VAD signals
        if not self._settings.vad_signals:
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                if self._socket_client:
                    await self._socket_client.flush()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, validate, sync state, and reconnect.

        Args:
            delta: A :class:`STTSettings` (or ``SarvamSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.

        Raises:
            ValueError: If a setting is not supported by the current model.
        """
        # Validate against model capabilities before applying
        if is_given(delta.language) and delta.language is not None:
            if not self._config.supports_language:
                raise ValueError(
                    f"Model '{self._settings.model}' does not support language parameter "
                    "(auto-detects language)."
                )

        changed = await super()._update_settings(delta)

        # These are all WebSocket connect-time parameters; reconnect to apply.
        reconnect_fields = {
            "language",
            "positive_speech_threshold",
            "negative_speech_threshold",
            "min_speech_frames",
            "first_turn_min_speech_frames",
            "negative_frames_count",
            "negative_frames_window",
            "start_speech_volume_threshold",
            "interrupt_min_speech_frames",
            "pre_speech_pad_frames",
            "num_initial_ignored_frames",
        }
        if changed.keys() & reconnect_fields:
            await self._disconnect()
            await self._connect()

        unhandled = {k: v for k, v in changed.items() if k not in reconnect_fields}
        if unhandled:
            self._warn_unhandled_updated_settings(unhandled)

        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio data to Sarvam for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        if not self._socket_client:
            yield None
            return

        try:
            # Convert audio bytes to base64 for Sarvam API
            audio_base64 = base64.b64encode(audio).decode("utf-8")

            # Convert input_audio_codec to encoding format (prepend "audio/" if needed)
            encoding = (
                self._input_audio_codec
                if self._input_audio_codec.startswith("audio/")
                else f"audio/{self._input_audio_codec}"
            )

            # Build method arguments
            method_kwargs = {
                "audio": audio_base64,
                "encoding": encoding,
                "sample_rate": self.sample_rate,
            }

            await self._socket_client.transcribe(**method_kwargs)

        except Exception as e:
            yield ErrorFrame(error=f"Error sending audio to Sarvam: {e}", exception=e)

        yield None

    async def _connect(self):
        """Connect to Sarvam WebSocket API using the SDK."""
        logger.debug("Connecting to Sarvam")

        try:
            # Build common connection parameters
            connect_kwargs = {
                "model": self._settings.model,
                "sample_rate": str(self.sample_rate),
            }

            # Enable flush signal when using Pipecat's VAD (not Sarvam's) so that
            # the flush() call on user-stopped-speaking is honored by the server.
            if not self._settings.vad_signals:
                connect_kwargs["flush_signal"] = "true"

            # Only send vad parameters when explicitly set (avoid overriding server defaults)
            if self._settings.vad_signals is not None:
                connect_kwargs["vad_signals"] = "true" if self._settings.vad_signals else "false"
            if self._settings.high_vad_sensitivity is not None:
                connect_kwargs["high_vad_sensitivity"] = (
                    "true" if self._settings.high_vad_sensitivity else "false"
                )

            # Fine-grained VAD parameters (sent as strings per SDK spec)
            _vad_params = {
                "positive_speech_threshold": self._settings.positive_speech_threshold,
                "negative_speech_threshold": self._settings.negative_speech_threshold,
                "min_speech_frames": self._settings.min_speech_frames,
                "first_turn_min_speech_frames": self._settings.first_turn_min_speech_frames,
                "negative_frames_count": self._settings.negative_frames_count,
                "negative_frames_window": self._settings.negative_frames_window,
                "start_speech_volume_threshold": self._settings.start_speech_volume_threshold,
                "interrupt_min_speech_frames": self._settings.interrupt_min_speech_frames,
                "pre_speech_pad_frames": self._settings.pre_speech_pad_frames,
                "num_initial_ignored_frames": self._settings.num_initial_ignored_frames,
            }
            for k, v in _vad_params.items():
                if v is not None:
                    connect_kwargs[k] = str(v)

            # Add language_code for models that support it
            language_string = self._get_language_string()
            if language_string is not None:
                connect_kwargs["language_code"] = language_string

            # Add mode for models that support it
            if self._config.supports_mode and self._mode is not None:
                connect_kwargs["mode"] = self._mode

            # Headers are supplied through request_options because this is a
            # documented SDK parameter that survives SDK signature changes.
            request_options: RequestOptions = {"additional_headers": self._sdk_headers}

            try:
                self._websocket_context = self._sarvam_client.speech_to_text_streaming.connect(
                    **connect_kwargs,
                    request_options=request_options,
                )
            except TypeError:
                # Fallback for SDK builds that don't expose request_options.
                self._websocket_context = self._sarvam_client.speech_to_text_streaming.connect(
                    **connect_kwargs
                )

            # Enter the async context manager
            self._socket_client = await self._websocket_context.__aenter__()

            # Register event handler for incoming messages
            def _message_handler(message):
                """Wrapper to handle async response handler."""
                # Use Pipecat's built-in task management
                self.create_task(self._handle_message(message))

            self._socket_client.on(EventType.MESSAGE, _message_handler)

            # Start receive task using Pipecat's task management
            self._receive_task = self.create_task(self._receive_task_handler())

            self._create_keepalive_task()

            logger.info("Connected to Sarvam successfully")

        except ApiError as e:
            self._socket_client = None
            self._websocket_context = None
            await self.push_error(error_msg=f"Sarvam API error: {e}", exception=e)
        except Exception as e:
            self._socket_client = None
            self._websocket_context = None
            await self.push_error(error_msg=f"Failed to connect to Sarvam: {e}", exception=e)

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket API using SDK."""
        await self._cancel_keepalive_task()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        # Clear references first to prevent run_stt from sending audio
        # during the close handshake.
        socket_client = self._socket_client
        websocket_context = self._websocket_context
        self._socket_client = None
        self._websocket_context = None

        if websocket_context and socket_client:
            try:
                await websocket_context.__aexit__(None, None, None)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Error closing WebSocket connection: {e}", exception=e
                )
            finally:
                logger.debug("Disconnected from Sarvam WebSocket")

    async def _receive_task_handler(self):
        """Handle incoming messages from Sarvam WebSocket.

        This task wraps the SDK's start_listening() method which processes
        messages via the registered event handler callback.
        """
        if not self._socket_client:
            return

        try:
            # Start listening for messages from the Sarvam SDK
            # Messages will be handled via the _message_handler callback
            await self._socket_client.start_listening()
        except Exception as e:
            await self.push_error(error_msg=f"Sarvam receive task error: {e}", exception=e)

    async def _handle_message(self, message):
        """Handle incoming WebSocket message from Sarvam SDK.

        Processes transcription data and VAD events from the Sarvam service.

        Args:
            message: The parsed response object from Sarvam WebSocket.
        """
        logger.debug(f"Received response: {message}")

        try:
            if message.type == "events":
                # VAD event
                signal = message.data.signal_type
                timestamp = message.data.occured_at
                logger.debug(f"VAD Signal: {signal}, Occurred at: {timestamp}")

                if signal == "START_SPEECH":
                    logger.debug("User started speaking")
                    await self._call_event_handler("on_speech_started")
                    await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

                elif signal == "END_SPEECH":
                    logger.debug("User stopped speaking")
                    await self._call_event_handler("on_speech_stopped")
                    await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

            elif message.type == "data":
                transcript = message.data.transcript
                language_code = message.data.language_code
                # Prefer language from message (auto-detected for translate models). Fallback to configured.
                if language_code:
                    language = self._map_language_code_to_enum(language_code)
                else:
                    language_string = self._get_language_string()
                    if language_string:
                        language = self._map_language_code_to_enum(language_string)
                    else:
                        language = Language.HI_IN

                # Emit utterance end event
                await self._call_event_handler("on_utterance_end")

                if transcript and transcript.strip():
                    # Record tracing for this transcription event
                    await self._handle_transcription(transcript, True, language)
                    # Report usage before the transcription frame so tracing
                    # can attach it to the STT span the frame closes.
                    await self.emit_stt_usage_metrics()
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            language,
                            result=(message.dict() if hasattr(message, "dict") else str(message)),
                        )
                    )
        except Exception as e:
            await self.push_error(error_msg=f"Failed to handle message: {e}", exception=e)
            await self.stop_all_metrics()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing.

        This method is decorated with @traced_stt for observability.
        """
        pass

    def _map_language_code_to_enum(self, language_code: str) -> Language:
        """Map Sarvam language code to pipecat Language enum."""
        mapping = {
            "bn-IN": Language.BN_IN,
            "gu-IN": Language.GU_IN,
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "ml-IN": Language.ML_IN,
            "mr-IN": Language.MR_IN,
            "ta-IN": Language.TA_IN,
            "te-IN": Language.TE_IN,
            "pa-IN": Language.PA_IN,
            "od-IN": Language.OR_IN,
            "en-US": Language.EN_US,
            "en-IN": Language.EN_IN,
            "as-IN": Language.AS_IN,
        }
        return mapping.get(language_code, Language.HI_IN)

    def _is_keepalive_ready(self) -> bool:
        """Check if the Sarvam SDK websocket client is connected."""
        return self._socket_client is not None

    async def _send_keepalive(self, silence: bytes):
        """Send silent audio via the Sarvam SDK to keep the connection alive.

        Args:
            silence: Silent 16-bit mono PCM audio bytes.
        """
        audio_base64 = base64.b64encode(silence).decode("utf-8")
        encoding = (
            self._input_audio_codec
            if self._input_audio_codec.startswith("audio/")
            else f"audio/{self._input_audio_codec}"
        )
        method_kwargs = {
            "audio": audio_base64,
            "encoding": encoding,
            "sample_rate": self.sample_rate,
        }
        # We know client exists because _is_keepalive_ready(), called before
        # _send_keepalive(), gates on it
        assert self._socket_client is not None

        await self._socket_client.transcribe(**method_kwargs)


_REALTIME_MODEL = "saaras:v3-realtime"

SUPPORTED_LANGUAGES = {
    "auto",
    "en-IN",
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "or-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "gu-IN",
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
}
# A plain set: the rate in force can come from the pipeline rather than the
# caller, so it is checked once resolved rather than annotated.
SUPPORTED_SAMPLE_RATES = {8000, 16000}

# Sarvam's `stream_type` selects the *server* flush profile; it says nothing
# about how often the client should send. Audio goes out on a fixed cadence so
# transcript latency doesn't scale with the chosen profile.
_CLIENT_CHUNK_MS = 50

# Accepted in a `config.update` message, and so the whole of `Settings`. Of
# these, `language_code`, `stream_type`, `mode`, and `prompt` are
# boundary-gated: the server defers them to the next utterance boundary.
_RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "language_code",
        "stream_type",
        "mode",
        "prompt",
        "threshold",
        "silence_duration_ms",
        "min_speech_duration_ms",
    }
)

# Only meaningful while the server is doing the endpointing.
_VAD_TUNING_FIELDS = (
    "threshold",
    "silence_duration_ms",
    "min_speech_duration_ms",
)
_SHORT_LANGUAGE_DEFAULTS = {
    language.split("-", maxsplit=1)[0]: language
    for language in SUPPORTED_LANGUAGES
    if language != "auto"
}


def language_to_sarvam_realtime_language(language: Language) -> str:
    """Convert a Language enum to Sarvam realtime's language code."""
    language_map = {
        Language.AS_IN: "as-IN",
        Language.BN_IN: "bn-IN",
        Language.EN_IN: "en-IN",
        Language.GU_IN: "gu-IN",
        Language.HI_IN: "hi-IN",
        Language.KN_IN: "kn-IN",
        Language.KOK_IN: "kok-IN",
        Language.MAI_IN: "mai-IN",
        Language.ML_IN: "ml-IN",
        Language.MR_IN: "mr-IN",
        Language.OR_IN: "or-IN",
        Language.PA_IN: "pa-IN",
        Language.SD_IN: "sd-IN",
        Language.TA_IN: "ta-IN",
        Language.TE_IN: "te-IN",
    }
    return resolve_language(language, language_map, use_base_code=False)


@dataclass
class SarvamRealtimeSTTSettings(STTSettings):
    """Settings for SarvamRealtimeSTTService.

    Sarvam reads these from the connection query string but also accepts them
    in a ``config.update``. The values it only reads at connection time are
    constructor arguments on the service instead.

    Parameters:
        language_code: Sarvam realtime language code or ``auto``.
        stream_type: Streaming cadence: ``fast``, ``balanced``, or ``simulated``.
        mode: Realtime STT task mode.
        prompt: Optional decoding prompt.
        threshold: Optional VAD sensitivity threshold.
        silence_duration_ms: Optional silence duration for end-of-speech.
        min_speech_duration_ms: Optional minimum speech duration.
    """

    language_code: str | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stream_type: Literal["fast", "balanced", "simulated"] | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    mode: SarvamMode | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prompt: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    silence_duration_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_speech_duration_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SarvamRealtimeSTTService(WebsocketSTTService):
    """Sarvam realtime Speech-to-Text service.

    Streams raw audio bytes to Sarvam's realtime websocket endpoint and maps
    provider VAD and transcript events into Pipecat frames.

    With the default ``endpointing="vad"`` the server drives turn boundaries. With
    ``endpointing="manual"`` the pipeline drives them instead, so the pipeline
    needs a turn strategy that emits ``VADUserStartedSpeakingFrame`` and
    ``VADUserStoppedSpeakingFrame``; without one, Sarvam never receives a boundary
    and never emits a final transcript. The mode is a constructor argument rather
    than a setting, since it decides which turn strategies the user aggregator
    runs and those are announced once, at startup.

    A VAD analyzer is required in either ``endpointing`` mode. Under ``vad`` it
    times transcription latency: TTFB is measured from
    ``VADUserStoppedSpeakingFrame``, which carries the stop delay needed to place
    the real end of speech, where Sarvam's own ``vad.speech_end`` arrives only
    after the server's silence window and would time a shorter interval than
    every other STT service reports. Under ``manual`` those same frames also mark
    the turn for Sarvam, reaching it as ``speech_start`` and ``speech_end``.
    """

    Settings = SarvamRealtimeSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws",
        endpointing: Literal["vad", "manual"] = "vad",
        sample_rate: int | None = None,
        return_timestamps: bool = False,
        prefix_padding_ms: int | None = None,
        settings: Settings | None = None,
        should_interrupt: bool = True,
        ttfs_p99_latency: float | None = SARVAM_REALTIME_TTFS_P99,
        **kwargs,
    ):
        """Initialize Sarvam realtime STT.

        Args:
            api_key: Sarvam API key.
            base_url: Realtime STT websocket endpoint.
            endpointing: Which side detects turn boundaries: ``vad`` for Sarvam's
                own detection, or ``manual`` for the pipeline's. Decides the turn
                strategies this service asks the user aggregator to run, so it is
                fixed for the life of the service. Defaults to ``vad``.
            sample_rate: Declared input audio sample rate, 8000 or 16000. ``None``
                adopts the pipeline's input rate.
            return_timestamps: Whether final transcripts should include segment
                offsets. Defaults to False.
            prefix_padding_ms: Optional VAD prefix padding, used only under
                ``endpointing="vad"``.
            settings: Runtime-updatable realtime settings.
            should_interrupt: Determine whether the bot should be interrupted when
                Sarvam detects user speech. Passed along to the user turn
                strategies this service recommends, which own the interruption; a
                user-supplied ``user_turn_strategies`` overrides the recommendation
                and this setting with it. Defaults to True.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
            **kwargs: Additional arguments passed to :class:`WebsocketSTTService`.
        """
        settings_fields = {setting.name for setting in fields(self.Settings)}
        direct_settings = sorted(settings_fields.intersection(kwargs))
        if direct_settings:
            names = ", ".join(direct_settings)
            raise TypeError(
                f"{names} must be passed via "
                "settings=SarvamRealtimeSTTService.Settings(...), not as constructor kwargs"
            )
        if "reconnect_on_error" in kwargs:
            raise TypeError(
                "SarvamRealtimeSTTService does not support reconnect_on_error; "
                "reconnection is always disabled"
            )
        if sample_rate is not None and sample_rate not in SUPPORTED_SAMPLE_RATES:
            allowed = ", ".join(str(rate) for rate in sorted(SUPPORTED_SAMPLE_RATES))
            raise ValueError(f"Unsupported sample_rate '{sample_rate}'. Allowed values: {allowed}.")
        default_settings = self.Settings(
            model=_REALTIME_MODEL,
            language=None,
            language_code="en-IN",
            stream_type="balanced",
            mode="transcribe",
            prompt=None,
            threshold=None,
            silence_duration_ms=None,
            min_speech_duration_ms=None,
        )
        language_code_given = settings is not None and is_given(settings.language_code)
        if settings is not None:
            default_settings.apply_update(settings)

        # An explicit language_code wins; otherwise derive it from `language`.
        # `language` may still be a raw string here, since the base class only
        # normalizes it once super().__init__() runs.
        if not language_code_given:
            language = _as_language(default_settings.language)
            if language is not None:
                default_settings.language_code = language_to_sarvam_realtime_language(language)

        self._validate_settings(default_settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            reconnect_on_error=False,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._return_timestamps = return_timestamps
        self._prefix_padding_ms = prefix_padding_ms
        self._should_interrupt = should_interrupt
        self._receive_task: asyncio.Task | None = None
        self._audio_buffer = bytearray()
        self._request_id: str | None = None
        self._provider_speech_active = False
        self._speech_end_audio_position_s: float | None = None
        self._audio_position_bytes = 0
        self._endpointing = endpointing

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing and usage metrics."""
        return True

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Request external turn strategies when Sarvam endpoints server-side.

        With ``endpointing="vad"`` (the default) Sarvam's VAD decides turn
        boundaries and this service proposes them via
        ``ProposedUserStarted/StoppedSpeakingFrame``, so the user aggregator
        resolves those rather than running local VAD/smart-turn. In
        ``endpointing="manual"`` the pipeline supplies the boundaries, so the
        defaults are left in place. Applied unless the user passed their own
        ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        if self._endpointing == "vad":
            frame.user_turn_strategies = ExternalUserTurnStrategies(
                enable_interruptions=self._should_interrupt,
            )
        return frame

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Language enum to Sarvam realtime's language code."""
        return language_to_sarvam_realtime_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect the websocket.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        # The rate can come from the pipeline, so it is only settled once the
        # service is set up. It holds for the session, so an unsupported one is
        # a permanent failure that costs the service its usability, letting a
        # `ServiceSwitcher` move on.
        error = self._resolved_sample_rate_error()
        if error:
            await self.push_error(error, category=ErrorCategory.INVALID_REQUEST)
            return
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the websocket."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the websocket."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and send manual endpointing boundaries when configured."""
        await super().process_frame(frame, direction)
        if self._endpointing != "manual":
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._send_json({"event": "speech_start"})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # The tail of the turn must reach Sarvam before the boundary, or the
            # final transcript is cut short. `speech_end` finalizes the utterance
            # on its own; Sarvam's separate `flush` event force-finalizes
            # buffered audio mid-utterance and has no part in a turn that ends
            # on a boundary.
            await self._flush_audio_buffer()
            await self._send_json({"event": "speech_end"})

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Buffer and send raw audio bytes to Sarvam as base64-encoded JSON messages."""
        if not audio:
            yield None
            return
        if not self._is_websocket_open():
            yield None
            return

        self._audio_buffer.extend(audio)
        chunk_size = self._chunk_size_bytes()
        while len(self._audio_buffer) >= chunk_size:
            chunk = bytes(self._audio_buffer[:chunk_size])
            del self._audio_buffer[:chunk_size]
            try:
                await self._send_audio_chunk(chunk)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Sarvam realtime STT send failed: {e}", exception=e
                )
                break

        yield None

    def _build_ws_url(self) -> str:
        """Build the Sarvam realtime websocket URL."""
        params = self._query_params()
        return f"{self._base_url}?{urlencode(params)}"

    async def _connect(self):
        """Connect to Sarvam realtime and start receive task."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Sarvam realtime."""
        await super()._disconnect()
        if self._websocket and self._websocket.state is State.OPEN:
            await self._flush_audio_buffer()
            try:
                await self._send_json({"event": "end"})
            except Exception as e:
                logger.debug(f"{self} error sending Sarvam end event: {e}")

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open the Sarvam realtime websocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            url = self._build_ws_url()
            headers = {"API-SUBSCRIPTION-KEY": self._api_key}
            logger.debug(f"Connecting to Sarvam realtime STT WebSocket: {url}")
            self._websocket = await self._websocket_connect(
                url,
                additional_headers=headers,
                user_agent_header=sdk_headers()["User-Agent"],
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            # Left on the category the failure earns: `_try_reconnect` skips a
            # service that has stopped being usable, so reporting a socket that
            # would not open as permanent bars the retry that could open it.
            await self.push_error(
                error_msg=f"Unable to connect to Sarvam realtime STT: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect_websocket(self):
        """Close the active websocket."""
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(
                error_msg=f"Error closing Sarvam realtime STT websocket: {e}", exception=e
            )
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_task_handler(self, report_error: ReportErrorCallback):
        """Close out the active utterance once the receive loop is done.

        Reconnection is disabled, so the loop exiting means no further server
        event can arrive. An utterance still open at that point would leave
        downstream turn aggregation waiting on a boundary that is never coming,
        and the service has no transcripts left to give, so it also stops being
        usable — the base class reports the drop as retryable, which holds only
        for services that reconnect on demand. Cancellation is left alone: that
        only happens during an intentional disconnect, where teardown is
        already under way.
        """
        await super()._receive_task_handler(report_error)
        await self._complete_active_utterance()
        if not self._disconnecting:
            await self.set_usable(False)

    async def _receive_messages(self):
        """Receive Sarvam realtime server events."""
        if not self._websocket:
            raise Exception("Websocket not connected")
        async for message in self._websocket:
            if not isinstance(message, str):
                logger.trace(f"{self} ignored non-text Sarvam server message")
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON Sarvam message: {message}")
                continue
            await self._handle_message(data)

    async def _handle_message(self, message: dict[str, Any]):
        """Handle a parsed Sarvam realtime server event."""
        event = message.get("event")
        if event == "session.begin":
            self._request_id = message.get("request_id")
            logger.info(f"{self} Sarvam realtime session.begin request_id={self._request_id}")
        # Only the endpointer in charge gets to set turn boundaries. Under
        # manual endpointing the pipeline owns them, so server VAD telemetry
        # would compete with the boundaries it is already producing.
        elif event == "vad.speech_start" and self._endpointing == "vad":
            await self._handle_speech_start(message)
        elif event == "vad.speech_end" and self._endpointing == "vad":
            await self._handle_speech_end(message)
        elif event == "transcript.partial":
            await self._handle_partial_transcript(message)
        elif event == "transcript.final":
            await self._handle_final_transcript(message)
        elif event == "session.end":
            await self._handle_session_end(message)
        elif event == "config.updated":
            logger.trace(f"{self} Sarvam realtime acknowledgement: {message}")
        elif event == "error":
            await self._handle_error(message)
        elif event == "pong":
            # Answers our keepalive ping; the reply itself is the liveness proof.
            logger.trace(f"{self} Sarvam realtime pong")
        else:
            logger.trace(f"{self} unhandled Sarvam realtime event: {message}")

    async def update_config(self, **fields: Any):
        """Send a live Sarvam ``config.update`` message without reconnecting.

        Args:
            fields: Runtime-updatable Sarvam config values. Connection-only
                values are rejected; they are fixed for the life of the stream.

        Raises:
            ValueError: If a connection-only or invalid value is supplied.
        """
        if not fields:
            return
        self._validate_config_update(fields)
        await self._send_config_update(fields)

    async def _send_config_update(self, fields: dict[str, Any]):
        """Send an already-validated ``config.update`` and record what was sent."""
        if not await self._send_json({"event": "config.update", **fields}):
            return
        # The store has to follow what the server was told, or a later delta
        # diffs against a stale value and skips an update the server needs.
        self._settings.apply_update(self.Settings(**fields))

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply runtime settings and send supported fields via ``config.update``."""
        delta = self._with_derived_language_code(delta)
        proposed = self._settings.copy()
        proposed.apply_update(delta)

        # Drawn from the runtime-updatable set, so the payload carries nothing
        # a `config.update` would have to reject.
        payload = {
            name: getattr(proposed, name)
            for name in _RUNTIME_CONFIG_FIELDS
            if is_given(getattr(proposed, name))
            and getattr(proposed, name) != getattr(self._settings, name)
        }

        changed = await super()._update_settings(delta)
        if not changed:
            return changed

        # `language` reaches Sarvam as `language_code`, so it is never unhandled.
        unsupported = set(changed) - _RUNTIME_CONFIG_FIELDS - {"language"}
        if unsupported:
            self._warn_unhandled_updated_settings({key: changed[key] for key in unsupported})

        if payload:
            await self._send_config_update(payload)
        return changed

    def _with_derived_language_code(self, delta: STTSettings) -> Settings:
        """Fill in ``language_code`` from a ``language`` delta.

        A caller can send the base :class:`STTSettings`, which carries
        ``language`` but none of the Sarvam fields, so the delta is widened to
        these settings first. Mirrors the constructor: an explicit
        ``language_code`` wins, since it also expresses ``auto``, which has no
        :class:`Language` equivalent.
        """
        if not isinstance(delta, self.Settings):
            delta = self.Settings.from_mapping(delta.given_fields())
        if is_given(delta.language_code):
            return delta
        language = _as_language(delta.language)
        if language is None:
            return delta
        derived = delta.copy()
        derived.language_code = language_to_sarvam_realtime_language(language)
        return derived

    async def _handle_speech_start(self, message: dict[str, Any]):
        if self._provider_speech_active:
            return
        self._provider_speech_active = True
        self._speech_end_audio_position_s = None
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def _handle_speech_end(self, message: dict[str, Any]):
        await self._complete_active_utterance()

    async def _complete_active_utterance(self):
        """End the in-flight utterance.

        Runs on ``vad.speech_end`` and on a session ending mid-utterance. Without
        the latter, downstream turn aggregation would wait forever for a boundary
        the server is never going to send.
        """
        if not self._provider_speech_active:
            return
        self._provider_speech_active = False
        self._speech_end_audio_position_s = self._duration_for_bytes(self._audio_position_bytes)
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _handle_partial_transcript(self, message: dict[str, Any]):
        text = (message.get("text") or "").strip()
        if not text:
            return
        result = self._result_payload(message)
        language = self._language_for_frame(message.get("language"))
        await self.push_frame(
            InterimTranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=result,
            )
        )

    async def _handle_final_transcript(self, message: dict[str, Any]):
        # Report usage before the transcription frame so tracing can attach it
        # to the STT span the frame closes. A blank final still consumed audio.
        await self.emit_stt_usage_metrics()
        text = (message.get("text") or "").strip()
        if text:
            language = self._language_for_frame(message.get("language"))
            result = self._result_payload(message)
            result["speech_end_audio_position_s"] = self._speech_end_audio_position_s
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=result,
                    finalized=True,
                )
            )
            await self._trace_transcription(text, True, language)

    async def _handle_session_end(self, message: dict[str, Any]):
        if message.get("request_id"):
            self._request_id = message.get("request_id")
        await self._complete_active_utterance()

    async def _handle_error(self, message: dict[str, Any]):
        await self.push_error(
            error_msg=f"Sarvam realtime STT error: {json.dumps(message, ensure_ascii=False)}",
        )

    async def _send_audio_chunk(self, chunk: bytes):
        if not self._websocket:
            raise RuntimeError("WebSocket not connected")
        sent = await self._send_json(
            {"event": "audio_input", "audio": base64.b64encode(chunk).decode("utf-8")}
        )
        if not sent:
            return
        self._audio_position_bytes += len(chunk)

    async def _flush_audio_buffer(self):
        if not self._audio_buffer:
            return
        chunk = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        if not self._is_websocket_open():
            return
        try:
            await self._send_audio_chunk(chunk)
        except Exception as e:
            # Late audio on a socket the server already dropped must not break
            # the turn boundary or teardown.
            await self.push_error(error_msg=f"Sarvam realtime STT flush failed: {e}", exception=e)

    async def _send_keepalive(self, silence: bytes):
        """Hold the connection open with Sarvam's ping event.

        Args:
            silence: Silent PCM audio bytes, unused. The socket only accepts
                JSON events, and Sarvam answers a ping with a pong rather than
                reading padding audio.
        """
        await self._send_json({"event": "ping"})

    async def _send_json(self, payload: dict[str, Any]) -> bool:
        """Send a JSON event, reporting whether it reached the socket."""
        if not self._is_websocket_open():
            return False
        assert self._websocket is not None
        await self._websocket.send(json.dumps(payload))
        return True

    def _query_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "language_code": self._settings.language_code,
            "stream_type": self._settings.stream_type,
            "endpointing": self._endpointing,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
            "model": self._settings.model,
            "mode": self._settings.mode,
            "return_timestamps": str(self._return_timestamps).lower(),
        }
        optional: dict[str, Any] = {"prompt": self._settings.prompt}
        if self._endpointing == "vad":
            optional["prefix_padding_ms"] = self._prefix_padding_ms
            optional.update(
                {name: getattr(self._settings, name) for name in _VAD_TUNING_FIELDS},
            )
        params.update(
            {key: value for key, value in optional.items() if is_given(value) and value is not None}
        )
        return params

    def _bytes_per_second(self) -> int:
        return self.sample_rate * 2

    def _chunk_size_bytes(self) -> int:
        return int(self._bytes_per_second() * (_CLIENT_CHUNK_MS / 1000))

    def _duration_for_bytes(self, byte_count: int) -> float:
        bytes_per_second = self._bytes_per_second()
        if bytes_per_second <= 0:
            return 0.0
        return byte_count / bytes_per_second

    def _result_payload(self, message: dict[str, Any]) -> dict[str, Any]:
        payload = dict(message)
        if self._request_id is not None:
            payload.setdefault("request_id", self._request_id)
        confidence = payload.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            # `language_confidence` is a LID score and never a recognition score.
            payload["confidence"] = 1.0
        return payload

    def _language_for_frame(self, raw_language: str | None = None) -> Language | None:
        configured_language_code = assert_given(self._settings.language_code)
        language_code = self._normalize_language_code(raw_language or configured_language_code)
        if language_code == "auto":
            return None
        try:
            return Language(language_code)
        except ValueError:
            return None

    def _normalize_language_code(self, language_code: str | None) -> str:
        if not language_code:
            return assert_given(self._settings.language_code)
        if "-" not in language_code and language_code != "auto":
            configured = assert_given(self._settings.language_code)
            if configured != "auto" and configured.startswith(f"{language_code}-"):
                return configured
            return _SHORT_LANGUAGE_DEFAULTS.get(language_code, language_code)
        return language_code

    def _is_websocket_open(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    def _resolved_sample_rate_error(self) -> str | None:
        """Describe why the rate actually in use is unusable, if it is."""
        if self.sample_rate in SUPPORTED_SAMPLE_RATES:
            return None
        allowed = ", ".join(str(rate) for rate in sorted(SUPPORTED_SAMPLE_RATES))
        return f"Unsupported sample_rate '{self.sample_rate}'. Allowed values: {allowed}."

    def _validate_config_update(self, update: dict[str, Any]):
        """Reject a ``config.update`` field Sarvam has no setting for."""
        unknown = sorted(set(update) - {setting.name for setting in fields(self.Settings)})
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"Unknown config.update field(s) {names}.")

    @staticmethod
    def _validate_settings(settings: Settings):
        """Check the settings this integration itself depends on.

        Sarvam's own vocabulary — languages, modes, stream types, VAD tuning
        ranges — is left to the server, which rejects a bad value on the wire
        and reaches the app as an error frame. Repeating those lists here would
        block values Sarvam adds later.
        """
        model = assert_given(settings.model)
        if model != _REALTIME_MODEL:
            raise ValueError(f"Unsupported model '{model}'. Only '{_REALTIME_MODEL}' is supported.")

    @traced_stt
    async def _trace_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Record transcription event for tracing."""
        pass


def _as_language(value: Any) -> Language | None:
    """Coerce a settings ``language`` value to a ``Language``, or ``None``."""
    if isinstance(value, Language):
        return value
    if not isinstance(value, str):
        return None
    try:
        return Language(value)
    except ValueError:
        return None
