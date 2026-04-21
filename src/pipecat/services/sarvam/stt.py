#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam AI Speech-to-Text service implementation.

This module provides a streaming Speech-to-Text service using Sarvam AI's WebSocket-based
API. It supports real-time transcription with Voice Activity Detection (VAD) and
can handle multiple audio formats for Indian language speech recognition.
"""

import base64
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import (
    NOT_GIVEN,
    STTSettings,
    _NotGiven,
    is_given,
)
from pipecat.services.stt_latency import SARVAM_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from sarvamai import AsyncSarvamAI
    from sarvamai.core.api_error import ApiError
    from sarvamai.core.events import EventType
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sarvam, you need to `pip install pipecat-ai[sarvam]`.")
    raise Exception(f"Missing module: {e}")


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


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for a Sarvam STT model.

    Parameters:
        supports_prompt: Whether the model accepts prompt parameter.
        supports_mode: Whether the model accepts mode parameter.
        supports_language: Whether the model accepts language parameter.
        supports_vad_params: Whether the model accepts fine-grained VAD parameters.
        default_language: Default language code (None = auto-detect).
        default_mode: Default mode (None = not applicable).
        use_translate_endpoint: Whether to use speech_to_text_translate_streaming endpoint.
        use_translate_method: Whether to use translate() method instead of transcribe().
    """

    supports_prompt: bool
    supports_mode: bool
    supports_language: bool
    supports_vad_params: bool
    default_language: str | None
    default_mode: str | None
    use_translate_endpoint: bool
    use_translate_method: bool


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "saarika:v2.5": ModelConfig(
        supports_prompt=False,
        supports_mode=False,
        supports_language=True,
        supports_vad_params=False,
        default_language="unknown",
        default_mode=None,
        use_translate_endpoint=False,
        use_translate_method=False,
    ),
    "saaras:v2.5": ModelConfig(
        supports_prompt=True,
        supports_mode=False,
        supports_language=False,
        supports_vad_params=False,
        default_language=None,  # Auto-detects language
        default_mode=None,
        use_translate_endpoint=True,
        use_translate_method=True,
    ),
    "saaras:v3": ModelConfig(
        supports_prompt=False,
        supports_mode=True,
        supports_language=True,
        supports_vad_params=True,
        default_language="unknown",
        default_mode="transcribe",
        use_translate_endpoint=False,
        use_translate_method=False,
    ),
}


@dataclass
class SarvamSTTSettings(STTSettings):
    """Settings for SarvamSTTService.

    Parameters:
        prompt: Optional prompt to guide transcription/translation style/context.
            Only applicable to models that support prompts (e.g., saaras:v2.5).
        vad_signals: Enable VAD signals in response.
        high_vad_sensitivity: Enable high VAD sensitivity.
        positive_speech_threshold: VAD probability threshold (0.0-1.0) above which
            a frame is considered speech. Only for saaras:v3.
        negative_speech_threshold: VAD probability threshold (0.0-1.0) below which
            a frame is considered silence. Only for saaras:v3.
        min_speech_frames: Minimum consecutive speech frames to start a speech
            segment. Only for saaras:v3.
        first_turn_min_speech_frames: Minimum speech frames for the first user
            turn. Only for saaras:v3.
        negative_frames_count: Number of silence frames within the window to end
            a speech segment. Only for saaras:v3.
        negative_frames_window: Sliding window size (in frames) for counting
            negative frames. Only for saaras:v3.
        start_speech_volume_threshold: Volume level (dB) below which audio is
            too quiet to be speech. Only for saaras:v3.
        interrupt_min_speech_frames: Minimum speech frames to register a
            barge-in/interruption. Only for saaras:v3.
        pre_speech_pad_frames: Number of audio frames to prepend before detected
            speech onset. Only for saaras:v3.
        num_initial_ignored_frames: Number of leading audio frames to skip at
            connection start. Only for saaras:v3.
    """

    prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_signals: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    high_vad_sensitivity: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    positive_speech_threshold: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    negative_speech_threshold: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_speech_frames: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    first_turn_min_speech_frames: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    negative_frames_count: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    negative_frames_window: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    start_speech_volume_threshold: float | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    interrupt_min_speech_frames: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pre_speech_pad_frames: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    num_initial_ignored_frames: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


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

    class InputParams(BaseModel):
        """Configuration parameters for Sarvam STT service.

        .. deprecated:: 0.0.105
            Use ``settings=SarvamSTTService.Settings(...)`` instead.

        Parameters:
            language: Target language for transcription.
                - saarika:v2.5: Defaults to "unknown" (auto-detect supported)
                - saaras:v2.5: Not used (auto-detects language)
                - saaras:v3: Defaults to "unknown" (auto-detect supported)
            prompt: Optional prompt to guide transcription/translation style/context.
                Only applicable to saaras:v2.5. Defaults to None.
            mode: Mode of operation for saaras:v3 models only. Options: transcribe, translate,
                verbatim, translit, codemix. Defaults to "transcribe" for saaras:v3.
            vad_signals: Enable VAD signals in response. Defaults to None.
            high_vad_sensitivity: Enable high VAD sensitivity. Defaults to None.
        """

        language: Language | None = None
        prompt: str | None = None
        mode: Literal["transcribe", "translate", "verbatim", "translit", "codemix"] | None = None
        vad_signals: bool | None = None
        high_vad_sensitivity: bool | None = None

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        mode: Literal["transcribe", "translate", "verbatim", "translit", "codemix"] | None = None,
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

            mode: Mode of operation. Options: transcribe, translate, verbatim,
                translit, codemix. Only applicable to models that support it
                (e.g., saaras:v3). Defaults to the model's default mode.
            sample_rate: Audio sample rate. Defaults to 16000 if not specified.
            input_audio_codec: Audio codec/format of the input file. Defaults to "wav".
            params: Configuration parameters for Sarvam STT service.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamSTTService.Settings(...)`` instead.

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
            model="saaras:v3",
            language=None,
            prompt=None,
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
                default_settings.prompt = params.prompt
                if params.mode is not None:
                    mode = params.mode
                default_settings.vad_signals = params.vad_signals
                default_settings.high_vad_sensitivity = params.high_vad_sensitivity

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        # Resolve model config and validate (after all overrides)
        resolved_model = default_settings.model
        if resolved_model not in MODEL_CONFIGS:
            allowed = ", ".join(sorted(MODEL_CONFIGS.keys()))
            raise ValueError(f"Unsupported model '{resolved_model}'. Allowed values: {allowed}.")

        self._config = MODEL_CONFIGS[resolved_model]

        # Validate parameters against model capabilities
        if default_settings.prompt is not None and not self._config.supports_prompt:
            raise ValueError(f"Model '{resolved_model}' does not support prompt parameter.")
        if mode is not None and not self._config.supports_mode:
            raise ValueError(f"Model '{resolved_model}' does not support mode parameter.")
        if default_settings.language is not None and not self._config.supports_language:
            raise ValueError(
                f"Model '{resolved_model}' does not support language parameter (auto-detects language)."
            )

        if not self._config.supports_vad_params:
            vad_param_names = [
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
            ]
            for param_name in vad_param_names:
                if getattr(default_settings, param_name) is not None:
                    raise ValueError(
                        f"Model '{resolved_model}' does not support {param_name} parameter. "
                        f"Fine-grained VAD parameters are only supported by saaras:v3."
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
        if self._settings.language:
            return language_to_sarvam_language(self._settings.language)
        return self._config.default_language

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames.

        Handles VAD frames for TTFB tracking when using Pipecat's VAD
        instead of Sarvam's built-in VAD.
        """
        await super().process_frame(frame, direction)

        # Only handle VAD frames when not using Sarvam's VAD signals
        if not self._settings.vad_signals:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                await self._start_metrics()
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
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
        if isinstance(delta, self.Settings) and is_given(delta.prompt) and delta.prompt is not None:
            if not self._config.supports_prompt:
                raise ValueError(
                    f"Model '{self._settings.model}' does not support prompt parameter."
                )

        if isinstance(delta, self.Settings) and not self._config.supports_vad_params:
            vad_param_names = [
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
            ]
            for param_name in vad_param_names:
                val = getattr(delta, param_name, NOT_GIVEN)
                if is_given(val) and val is not None:
                    raise ValueError(
                        f"Model '{self._settings.model}' does not support {param_name} "
                        f"parameter. Fine-grained VAD parameters are only supported by saaras:v3."
                    )

        changed = await super()._update_settings(delta)

        # These are all WebSocket connect-time parameters; reconnect to apply.
        reconnect_fields = {
            "language",
            "prompt",
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

    async def set_prompt(self, prompt: str | None):
        """Set the transcription/translation prompt and reconnect.

        .. deprecated:: 0.0.104
            Use ``STTUpdateSettingsFrame(SarvamSTTService.Settings(prompt=...))`` instead.

        Args:
            prompt: Prompt text to guide transcription/translation style/context.
                   Pass None to clear/disable prompt.
                   Only applicable to models that support prompts.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                f"{self.__class__.__name__}.set_prompt() is deprecated. "
                "Use STTUpdateSettingsFrame(self.Settings(prompt=...)) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not self._config.supports_prompt:
            if prompt is not None:
                raise ValueError(
                    f"Model '{self._settings.model}' does not support prompt parameter."
                )
            # If prompt is None and model doesn't support prompts, silently return (no-op)
            return

        logger.info(f"Updating {self._settings.model} prompt.")
        self._settings.prompt = prompt
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        """Start the Sarvam STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
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

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
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

            # Use appropriate method based on model configuration
            if self._config.use_translate_method:
                await self._socket_client.translate(**method_kwargs)
            else:
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

            # Fine-grained VAD parameters (saaras:v3 only, sent as strings per SDK spec)
            if self._config.supports_vad_params:
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

            # Prompt support differs across sarvamai versions. Prefer connect-time prompt
            # when available and gracefully degrade if the SDK doesn't accept it.
            if self._settings.prompt is not None and self._config.supports_prompt:
                connect_kwargs["prompt"] = self._settings.prompt

            def _connect_with_sdk_headers(connect_fn, **kwargs):
                # If prompt is unsupported at connect-time, retry without it.
                # Headers are supplied through request_options because this is a
                # documented SDK parameter that survives SDK signature changes.
                request_options = {"additional_headers": self._sdk_headers}

                attempts = [kwargs]
                if "prompt" in kwargs:
                    attempts.append({k: v for k, v in kwargs.items() if k != "prompt"})

                last_type_error = None
                for attempt_kwargs in attempts:
                    try:
                        return connect_fn(
                            **attempt_kwargs,
                            request_options=request_options,
                        )
                    except TypeError as e:
                        last_type_error = e
                    try:
                        # Fallback for SDK builds that don't expose request_options.
                        return connect_fn(**attempt_kwargs)
                    except TypeError as e:
                        last_type_error = e

                if last_type_error is not None:
                    raise last_type_error
                return connect_fn(**kwargs)

            # Choose the appropriate endpoint based on model configuration
            if self._config.use_translate_endpoint:
                self._websocket_context = _connect_with_sdk_headers(
                    self._sarvam_client.speech_to_text_translate_streaming.connect,
                    **connect_kwargs,
                )
            else:
                self._websocket_context = _connect_with_sdk_headers(
                    self._sarvam_client.speech_to_text_streaming.connect,
                    **connect_kwargs,
                )

            # Enter the async context manager
            self._socket_client = await self._websocket_context.__aenter__()

            # Fallback for SDKs that support runtime prompt updates.
            if self._settings.prompt is not None and self._config.supports_prompt:
                prompt_setter = getattr(self._socket_client, "set_prompt", None)
                if callable(prompt_setter):
                    await prompt_setter(self._settings.prompt)

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
                    await self._start_metrics()
                    logger.debug("User started speaking")
                    await self._call_event_handler("on_speech_started")
                    await self.broadcast_frame(UserStartedSpeakingFrame)
                    await self.broadcast_interruption()

                elif signal == "END_SPEECH":
                    logger.debug("User stopped speaking")
                    await self._call_event_handler("on_speech_stopped")
                    await self.broadcast_frame(UserStoppedSpeakingFrame)

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
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            language,
                            result=(message.dict() if hasattr(message, "dict") else str(message)),
                        )
                    )

                await self.stop_processing_metrics()

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
        if self._config.use_translate_method:
            await self._socket_client.translate(**method_kwargs)
        else:
            await self._socket_client.transcribe(**method_kwargs)

    async def _start_metrics(self):
        """Start processing metrics collection."""
        await self.start_processing_metrics()
