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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Literal, Optional

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
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, is_given
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

    Attributes:
        supports_prompt: Whether the model accepts prompt parameter.
        supports_mode: Whether the model accepts mode parameter.
        supports_language: Whether the model accepts language parameter.
        default_language: Default language code (None = auto-detect).
        default_mode: Default mode (None = not applicable).
        use_translate_endpoint: Whether to use speech_to_text_translate_streaming endpoint.
        use_translate_method: Whether to use translate() method instead of transcribe().
    """

    supports_prompt: bool
    supports_mode: bool
    supports_language: bool
    default_language: Optional[str]
    default_mode: Optional[str]
    use_translate_endpoint: bool
    use_translate_method: bool


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "saarika:v2.5": ModelConfig(
        supports_prompt=False,
        supports_mode=False,
        supports_language=True,
        default_language="unknown",
        default_mode=None,
        use_translate_endpoint=False,
        use_translate_method=False,
    ),
    "saaras:v2.5": ModelConfig(
        supports_prompt=True,
        supports_mode=False,
        supports_language=False,
        default_language=None,  # Auto-detects language
        default_mode=None,
        use_translate_endpoint=True,
        use_translate_method=True,
    ),
    "saaras:v3": ModelConfig(
        supports_prompt=False,
        supports_mode=True,
        supports_language=True,
        default_language="unknown",
        default_mode="transcribe",
        use_translate_endpoint=False,
        use_translate_method=False,
    ),
}


@dataclass
class SarvamSTTSettings(STTSettings):
    """Settings for the Sarvam STT service.

    Parameters:
        prompt: Optional prompt to guide transcription/translation style.
        mode: Mode of operation (transcribe, translate, verbatim, etc.).
        vad_signals: Enable VAD signals in response.
        high_vad_sensitivity: Enable high VAD sensitivity.
    """

    prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    mode: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_signals: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    high_vad_sensitivity: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


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

    _settings: SarvamSTTSettings

    class InputParams(BaseModel):
        """Configuration parameters for Sarvam STT service.

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
            high_vad_sensitivity: Enable high VAD (Voice Activity Detection) sensitivity. Defaults to None.
        """

        language: Optional[Language] = None
        prompt: Optional[str] = None
        mode: Optional[Literal["transcribe", "translate", "verbatim", "translit", "codemix"]] = None
        vad_signals: Optional[bool] = None
        high_vad_sensitivity: Optional[bool] = None

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saarika:v2.5",
        sample_rate: Optional[int] = None,
        input_audio_codec: str = "wav",
        params: Optional[InputParams] = None,
        ttfs_p99_latency: Optional[float] = SARVAM_TTFS_P99,
        keepalive_timeout: Optional[float] = None,
        keepalive_interval: float = 5.0,
        **kwargs,
    ):
        """Initialize the Sarvam STT service.

        Args:
            api_key: Sarvam API key for authentication.
            model: Sarvam model to use for transcription. Allowed values:
                - "saarika:v2.5": Standard STT model
                - "saaras:v2.5": STT-Translate model (auto-detects language, supports prompts)
                - "saaras:v3": Advanced STT model (supports mode)
            sample_rate: Audio sample rate. Defaults to 16000 if not specified.
            input_audio_codec: Audio codec/format of the input file. Defaults to "wav".
            params: Configuration parameters for Sarvam STT service.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            keepalive_timeout: Seconds of no audio before sending silence to keep the
                connection alive. None disables keepalive.
            keepalive_interval: Seconds between idle checks when keepalive is enabled.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        params = params or SarvamSTTService.InputParams()

        # Get model configuration (validates model exists)
        if model not in MODEL_CONFIGS:
            allowed = ", ".join(sorted(MODEL_CONFIGS.keys()))
            raise ValueError(f"Unsupported model '{model}'. Allowed values: {allowed}.")

        self._config = MODEL_CONFIGS[model]

        # Validate parameters against model capabilities
        if params.prompt is not None and not self._config.supports_prompt:
            raise ValueError(f"Model '{model}' does not support prompt parameter.")
        if params.mode is not None and not self._config.supports_mode:
            raise ValueError(f"Model '{model}' does not support mode parameter.")
        if params.language is not None and not self._config.supports_language:
            raise ValueError(
                f"Model '{model}' does not support language parameter (auto-detects language)."
            )

        # Resolve mode default from model config
        mode = params.mode if params.mode is not None else self._config.default_mode

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=keepalive_timeout,
            keepalive_interval=keepalive_interval,
            settings=SarvamSTTSettings(
                model=model,
                language=params.language,
                prompt=params.prompt,
                mode=mode,
                vad_signals=params.vad_signals,
                high_vad_sensitivity=params.high_vad_sensitivity,
            ),
            **kwargs,
        )

        self._api_key = api_key

        # Store connection parameters
        self._input_audio_codec = input_audio_codec

        # Initialize Sarvam SDK client
        self._sdk_headers = sdk_headers()
        # NOTE: We avoid passing non-standard kwargs here because different sarvamai
        # versions expose different constructor signatures (static type checkers
        # complain otherwise). We instead inject headers best-effort below.
        self._sarvam_client = AsyncSarvamAI(api_subscription_key=api_key)
        for attr in ("default_headers", "_default_headers", "headers", "_headers"):
            d = getattr(self._sarvam_client, attr, None)
            if isinstance(d, dict):
                d.update(self._sdk_headers)
                break
        self._websocket_context = None
        self._socket_client = None
        self._receive_task = None

        if params.vad_signals:
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

    def _get_language_string(self) -> Optional[str]:
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
            delta: A :class:`STTSettings` (or ``SarvamSTTSettings``) delta.

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

        if isinstance(delta, SarvamSTTSettings):
            if is_given(delta.prompt) and delta.prompt is not None:
                if not self._config.supports_prompt:
                    raise ValueError(
                        f"Model '{self._settings.model}' does not support prompt parameter."
                    )
            if is_given(delta.mode) and delta.mode is not None:
                if not self._config.supports_mode:
                    raise ValueError(
                        f"Model '{self._settings.model}' does not support mode parameter."
                    )

        changed = await super()._update_settings(delta)

        # TODO: someday we could reconnect here to apply updated settings.
        # Code might look something like the below:
        # if not changed:
        #     return changed

        # await self._disconnect()
        # await self._connect()

        self._warn_unhandled_updated_settings(changed)

        return changed

    async def set_prompt(self, prompt: Optional[str]):
        """Set the transcription/translation prompt and reconnect.

        .. deprecated::
            Use ``STTUpdateSettingsFrame(SarvamSTTSettings(prompt=...))`` instead.

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
                "Use STTUpdateSettingsFrame(SarvamSTTSettings(prompt=...)) instead.",
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
            logger.warning("WebSocket not connected, cannot process audio")
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

            # Add language_code for models that support it
            language_string = self._get_language_string()
            if language_string is not None:
                connect_kwargs["language_code"] = language_string

            # Add mode for models that support it
            if self._config.supports_mode and self._settings.mode is not None:
                connect_kwargs["mode"] = self._settings.mode

            # Prompt support differs across sarvamai versions. Prefer connect-time prompt
            # when available and gracefully degrade if the SDK doesn't accept it.
            if self._settings.prompt is not None and self._config.supports_prompt:
                connect_kwargs["prompt"] = self._settings.prompt

            def _connect_with_sdk_headers(connect_fn, **kwargs):
                # Different SDK versions may use different kwarg names.
                # If prompt is unsupported at connect-time, retry without it.
                attempts = [kwargs]
                if "prompt" in kwargs:
                    attempts.append({k: v for k, v in kwargs.items() if k != "prompt"})

                last_type_error = None
                for attempt_kwargs in attempts:
                    for header_kw in ("headers", "additional_headers", "extra_headers"):
                        try:
                            return connect_fn(**attempt_kwargs, **{header_kw: self._sdk_headers})
                        except TypeError as e:
                            last_type_error = e
                    try:
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

        if self._websocket_context and self._socket_client:
            try:
                # Exit the async context manager
                await self._websocket_context.__aexit__(None, None, None)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Error closing WebSocket connection: {e}", exception=e
                )
            finally:
                logger.debug("Disconnected from Sarvam WebSocket")
                self._socket_client = None
                self._websocket_context = None

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
                    await self.push_interruption_task_frame_and_wait()

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
        self, transcript: str, is_final: bool, language: Optional[Language] = None
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
