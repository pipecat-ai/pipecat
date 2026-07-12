#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Whisper speech-to-text services with locally-downloaded models.

This module implements Whisper transcription using locally-downloaded models,
supporting both Faster Whisper and MLX Whisper backends for efficient inference.
"""

import asyncio
import platform
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from loguru import logger
from typing_extensions import override

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, assert_given
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Whisper, you need to `uv add "pipecat-ai[whisper]"`.')
    raise ImportError(f"Missing module: {e}") from e

# MLX Whisper only runs on Apple Silicon. On other platforms the package is
# unavailable (or installed but unloadable, e.g. a missing ``libmlx.so``), so
# importing it would break this module everywhere else. Only attempt it on macOS;
# WhisperSTTServiceMLX imports it lazily when actually used.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    try:
        import mlx_whisper  # noqa: F401
    except ModuleNotFoundError as e:
        logger.error(f"Exception: {e}")
        logger.error('In order to use Whisper, you need to `uv add "pipecat-ai[mlx-whisper]"`.')
        raise ImportError(f"Missing module: {e}") from e


class Model(StrEnum):
    """Whisper model selection options for Faster Whisper.

    Provides various model sizes and specializations for speech recognition,
    balancing quality and performance based on use case requirements.

    Parameters:
        TINY: Smallest multilingual model, fastest inference.
        BASE: Basic multilingual model, good speed/quality balance.
        SMALL: Small multilingual model, better speed/quality balance than BASE.
        MEDIUM: Medium-sized multilingual model, better quality.
        LARGE: Best quality multilingual model, slower inference.
        LARGE_V3_TURBO: Fast multilingual model, slightly lower quality than LARGE.
        DISTIL_LARGE_V2: Fast multilingual distilled model.
        DISTIL_MEDIUM_EN: Fast English-only distilled model.
    """

    # Multilingual models
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large-v3"
    LARGE_V3_TURBO = "deepdml/faster-whisper-large-v3-turbo-ct2"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"

    # English-only models
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"


class MLXModel(StrEnum):
    """MLX Whisper model selection options for Apple Silicon.

    Provides various model sizes optimized for Apple Silicon hardware,
    including quantized variants for improved performance.

    Parameters:
        TINY: Smallest multilingual model for MLX.
        MEDIUM: Medium-sized multilingual model for MLX.
        LARGE_V3: Best quality multilingual model for MLX.
        LARGE_V3_TURBO: Finetuned, pruned Whisper large-v3, much faster with slightly lower quality.
        DISTIL_LARGE_V3: Fast multilingual distilled model for MLX.
        LARGE_V3_TURBO_Q4: LARGE_V3_TURBO quantized to Q4 for reduced memory usage.
    """

    # Multilingual models
    TINY = "mlx-community/whisper-tiny"
    MEDIUM = "mlx-community/whisper-medium-mlx"
    LARGE_V3 = "mlx-community/whisper-large-v3-mlx"
    LARGE_V3_TURBO = "mlx-community/whisper-large-v3-turbo"
    DISTIL_LARGE_V3 = "mlx-community/distil-whisper-large-v3"
    LARGE_V3_TURBO_Q4 = "mlx-community/whisper-large-v3-turbo-q4"


def language_to_whisper_language(language: Language) -> str:
    """Maps pipecat Language enum to Whisper language codes.

    Args:
        language: A Language enum value representing the input language.

    Returns:
        The corresponding service language code. If ``language`` is not in
        the verified mapping, falls back to the base language code (e.g.,
        ``en`` from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).

    Note:
        Only includes languages officially supported by Whisper.
    """
    LANGUAGE_MAP = {
        # Arabic
        Language.AR: "ar",
        # Bengali
        Language.BN: "bn",
        # Czech
        Language.CS: "cs",
        # Danish
        Language.DA: "da",
        # German
        Language.DE: "de",
        # Greek
        Language.EL: "el",
        # English
        Language.EN: "en",
        # Spanish
        Language.ES: "es",
        # Persian
        Language.FA: "fa",
        # Finnish
        Language.FI: "fi",
        # French
        Language.FR: "fr",
        # Hindi
        Language.HI: "hi",
        # Hungarian
        Language.HU: "hu",
        # Indonesian
        Language.ID: "id",
        # Italian
        Language.IT: "it",
        # Japanese
        Language.JA: "ja",
        # Korean
        Language.KO: "ko",
        # Dutch
        Language.NL: "nl",
        # Polish
        Language.PL: "pl",
        # Portuguese
        Language.PT: "pt",
        # Romanian
        Language.RO: "ro",
        # Russian
        Language.RU: "ru",
        # Slovak
        Language.SK: "sk",
        # Swedish
        Language.SV: "sv",
        # Thai
        Language.TH: "th",
        # Turkish
        Language.TR: "tr",
        # Ukrainian
        Language.UK: "uk",
        # Urdu
        Language.UR: "ur",
        # Vietnamese
        Language.VI: "vi",
        # Chinese
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class WhisperSTTSettings(STTSettings):
    """Settings for WhisperSTTService.

    Parameters:
        no_speech_prob: Probability threshold for filtering non-speech segments.
    """

    no_speech_prob: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class WhisperMLXSTTSettings(STTSettings):
    """Settings for WhisperMLXSTTService.

    Parameters:
        no_speech_prob: Probability threshold for filtering non-speech segments.
        temperature: Sampling temperature (0.0-1.0).
        engine: Whisper engine identifier.
    """

    no_speech_prob: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    engine: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class WhisperStreamingSTTSettings(WhisperSTTSettings):
    """Settings for WhisperStreamingSTTService.

    Parameters:
        interim_model: Whisper model used for live interim transcriptions.
            Should be small enough to transcribe the buffered utterance well
            within ``interim_interval``.
        interim_interval: Target seconds between interim transcription passes.
    """

    interim_model: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interim_interval: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class WhisperSTTService(SegmentedSTTService):
    """Class to transcribe audio with a locally-downloaded Whisper model.

    This service uses Faster Whisper to perform speech-to-text transcription on audio
    segments. It supports multiple languages and various model sizes.
    """

    Settings = WhisperSTTSettings
    _settings: Settings

    @property
    def wants_wav_segments(self) -> bool:
        """Receive segments as raw 16-bit PCM, which the model reads directly."""
        return False

    def __init__(
        self,
        *,
        model: str | Model | None = None,
        device: str = "auto",
        compute_type: str = "default",
        no_speech_prob: float | None = None,
        language: Language | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Whisper STT service.

        Args:
            model: The Whisper model to use for transcription. Can be a Model enum or string.

                .. deprecated:: 0.0.105
                    Use ``settings=WhisperSTTService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            device: The device to run inference on ('cpu', 'cuda', or 'auto').
                Defaults to ``"auto"``.
            compute_type: The compute type for inference ('default', 'int8',
                'int8_float16', etc.). Defaults to ``"default"``.
            no_speech_prob: Probability threshold for filtering out non-speech segments.

                .. deprecated:: 0.0.105
                    Use ``settings=WhisperSTTService.Settings(no_speech_prob=...)`` instead.
                    Will be removed in 2.0.0.

            language: The default language for transcription.

                .. deprecated:: 0.0.105
                    Use ``settings=WhisperSTTService.Settings(language=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(
            model=Model.DISTIL_MEDIUM_EN.value,
            language=Language.EN,
            no_speech_prob=0.4,
        )

        # --- 2. Deprecated direct-arg overrides ---
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model if isinstance(model, str) else model.value
        if no_speech_prob is not None:
            self._warn_init_param_moved_to_settings("no_speech_prob", "no_speech_prob")
            default_settings.no_speech_prob = no_speech_prob
        if language is not None:
            self._warn_init_param_moved_to_settings("language", "language")
            default_settings.language = language

        # --- 3. (no params object for this service) ---

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            **kwargs,
        )

        # Init-only inference config
        self._device = device
        self._compute_type = compute_type

        self._model: WhisperModel | None = None

        self._load()

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert from pipecat Language to Whisper language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The corresponding Whisper language code, or None if not supported.
        """
        return language_to_whisper_language(language)

    def _load(self):
        """Loads the Whisper model.

        Note:
            If this is the first time this model is being run,
            it will take time to download from the Hugging Face model hub.
        """
        logger.debug("Loading Whisper model...")
        model_name = assert_given(self._settings.model)
        if model_name is None:
            raise ValueError("Whisper model must be specified")
        self._model = WhisperModel(model_name, device=self._device, compute_type=self._compute_type)
        logger.debug("Loaded Whisper model")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio data using Whisper.

        Args:
            audio: Raw audio bytes in 16-bit PCM format.

        Yields:
            Frame: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.

        Note:
            The audio is expected to be 16-bit signed PCM data.
            The service will normalize it to float32 in the range [-1, 1].
        """
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        language = assert_given(self._settings.language)
        segments, _ = await asyncio.to_thread(
            self._model.transcribe, audio_float, language=language
        )
        text: str = ""
        no_speech_prob_threshold = assert_given(self._settings.no_speech_prob)
        for segment in segments:
            if (
                no_speech_prob_threshold is not None
                and segment.no_speech_prob < no_speech_prob_threshold
            ):
                text += f"{segment.text} "

        await self.stop_processing_metrics()

        if text:
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
            )


class WhisperSTTServiceMLX(WhisperSTTService):
    """Subclass of `WhisperSTTService` with MLX Whisper model support.

    This service uses MLX Whisper to perform speech-to-text transcription on audio
    segments. It's optimized for Apple Silicon and supports multiple languages and quantizations.
    """

    Settings = WhisperMLXSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: str | MLXModel | None = None,
        no_speech_prob: float | None = None,
        language: Language | None = None,
        temperature: float | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the MLX Whisper STT service.

        Args:
            model: The MLX Whisper model to use for transcription. Can be an MLXModel enum or string.

                .. deprecated:: 0.0.105
                    Use ``settings=WhisperSTTServiceMLX.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            no_speech_prob: Probability threshold for filtering out non-speech segments.

                .. deprecated:: 0.0.105
                    Use ``settings=WhisperSTTServiceMLX.Settings(no_speech_prob=...)`` instead.
                    Will be removed in 2.0.0.

            language: The default language for transcription.

                .. deprecated:: 0.0.105
                    Use ``settings=WhisperSTTServiceMLX.Settings(language=...)`` instead.
                    Will be removed in 2.0.0.

            temperature: Temperature for sampling. Can be a float or tuple of floats.

                .. deprecated:: 0.0.105
                    Use ``settings=WhisperSTTServiceMLX.Settings(temperature=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(
            model=MLXModel.TINY.value,
            language=Language.EN,
            no_speech_prob=0.6,
            temperature=0.0,
            engine="mlx",
        )

        # --- 2. Deprecated direct-arg overrides ---
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model if isinstance(model, str) else model.value
        if no_speech_prob is not None:
            self._warn_init_param_moved_to_settings("no_speech_prob", "no_speech_prob")
            default_settings.no_speech_prob = no_speech_prob
        if language is not None:
            self._warn_init_param_moved_to_settings("language", "language")
            default_settings.language = language
        if temperature is not None:
            self._warn_init_param_moved_to_settings("temperature", "temperature")
            default_settings.temperature = temperature

        # --- 3. (no params object for this service) ---

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        # Skip WhisperSTTService.__init__ and call its parent directly
        SegmentedSTTService.__init__(
            self,
            settings=default_settings,
            **kwargs,
        )

        # No need to call _load() as MLX Whisper loads models on demand

    @override
    def _load(self):
        """MLX Whisper loads models on demand, so this is a no-op."""
        pass

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    @override
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio data using MLX Whisper.

        The audio is expected to be 16-bit signed PCM data.
        MLX Whisper will handle the conversion internally.

        Args:
            audio: Raw audio bytes in 16-bit PCM format.

        Yields:
            Frame: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.
        """
        try:
            # This will trigger an exception in case we want to use
            # WhisperMLXSTTService in a platform different than macOS.
            import mlx_whisper

            await self.start_processing_metrics()

            # Divide by 32768 because we have signed 16-bit data.
            audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            model_path = assert_given(self._settings.model)
            if model_path is None:
                raise ValueError("Whisper model must be specified")
            temperature = assert_given(self._settings.temperature)
            language = assert_given(self._settings.language)
            chunk = await asyncio.to_thread(
                mlx_whisper.transcribe,
                audio_float,
                path_or_hf_repo=model_path,
                temperature=temperature,
                language=language,
            )
            text: str = ""
            no_speech_prob_threshold = assert_given(self._settings.no_speech_prob)
            for segment in chunk.get("segments", []):
                # Drop likely hallucinations
                if segment.get("compression_ratio", None) == 0.5555555555555556:
                    continue

                if (
                    no_speech_prob_threshold is not None
                    and segment.get("no_speech_prob", 0.0) < no_speech_prob_threshold
                ):
                    text += f"{segment.get('text', '')} "

            if len(text.strip()) == 0:
                text = None

            await self.stop_processing_metrics()

            if text:
                await self._handle_transcription(text, True, language)
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                )

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


class WhisperStreamingSTTService(STTService):
    """Local Whisper STT with live interim transcriptions.

    Buffers audio while the user speaks (gated by VAD events) and periodically
    re-transcribes the buffered segment with a small interim model, pushing an
    :class:`~pipecat.frames.frames.InterimTranscriptionFrame` whenever the
    partial text changes. When the user stops speaking, the full segment is
    transcribed once with the final model and pushed as a finalized
    :class:`~pipecat.frames.frames.TranscriptionFrame`.

    Whisper has no incremental decoding mode, so interim results are produced
    by re-running the interim model over the whole buffered segment on each
    pass. Re-transcription cost grows with utterance length; keep
    ``interim_model`` small (the default is ``tiny``).

    Requires VAD to be enabled in the pipeline to function properly. Maintains
    a small audio buffer while the user is not speaking to account for the
    delay between actual speech start and VAD detection.

    Example::

        stt = WhisperStreamingSTTService(
            settings=WhisperStreamingSTTService.Settings(
                model=Model.SMALL,
                interim_model=Model.TINY,
                interim_interval=0.25,
            ),
            compute_type="int8",
        )
    """

    Settings = WhisperStreamingSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        device: str = "auto",
        compute_type: str = "default",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the streaming Whisper STT service.

        Args:
            device: The device to run inference on ('cpu', 'cuda', or 'auto').
                Both models run on the same device. Defaults to ``"auto"``.
            compute_type: The compute type for inference ('default', 'int8',
                'int8_float16', etc.). Defaults to ``"default"``.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to STTService.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(
            model=Model.DISTIL_MEDIUM_EN.value,
            interim_model=Model.TINY.value,
            interim_interval=0.25,
            language=Language.EN,
            no_speech_prob=0.4,
        )

        # --- 2/3. (new service: no deprecated params or params object) ---

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        # Init-only inference config
        self._device = device
        self._compute_type = compute_type

        self._model: WhisperModel | None = None
        self._interim_model: WhisperModel | None = None

        self._audio_buffer = bytearray()
        self._audio_buffer_size_1s = 0
        self._interim_min_bytes = 0
        self._interim_task: asyncio.Task | None = None
        self._last_interim_text = ""

        self._load()

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert from pipecat Language to Whisper language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The corresponding Whisper language code, or None if not supported.
        """
        return language_to_whisper_language(language)

    async def start(self, frame: StartFrame):
        """Start the service and size the audio buffers.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        # 16-bit mono PCM: sample_rate * 2 bytes per second of audio.
        self._audio_buffer_size_1s = self.sample_rate * 2
        # Skip interim passes on less than half a second of audio; Whisper
        # mostly hallucinates on shorter segments.
        self._interim_min_bytes = self.sample_rate

    async def stop(self, frame: EndFrame):
        """Stop the service and any in-flight interim transcription.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._cancel_interim_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and any in-flight interim transcription.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._cancel_interim_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, running transcription passes on VAD events.

        VAD events are handled here, after the base class has already pushed
        them downstream, so transcription work never delays their propagation.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Buffer audio for interim and final transcription.

        Replaces the base implementation (which runs ``run_stt`` on every
        chunk): a local batch model transcribes buffered segments, so audio
        is accumulated here and transcription is driven by VAD events.

        Args:
            frame: The audio frame to process.
            direction: The direction of frame processing.
        """
        if self._muted:
            return

        # UserAudioRawFrame contains a user_id (e.g. Daily, Livekit)
        if hasattr(frame, "user_id"):
            self._user_id = frame.user_id
        # AudioRawFrame does not have a user_id (e.g. SmallWebRTCTransport, websockets)
        else:
            self._user_id = ""

        # If the user is speaking the audio buffer will keep growing.
        self._audio_buffer += frame.audio

        # If the user is not speaking we keep just a little bit of audio to
        # compensate for the delay between speech start and VAD detection.
        if not self._user_speaking and len(self._audio_buffer) > self._audio_buffer_size_1s:
            discarded = len(self._audio_buffer) - self._audio_buffer_size_1s
            self._audio_buffer = self._audio_buffer[discarded:]

    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        await self._cancel_interim_task()
        self._last_interim_text = ""
        self._interim_task = self.create_task(self._interim_task_handler(), "interim_stt")

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        await self._cancel_interim_task()

        audio = bytes(self._audio_buffer)

        # Start clean.
        self._audio_buffer.clear()
        self._last_interim_text = ""

        await self.process_generator(self.run_stt(audio))

    async def _cancel_interim_task(self):
        if self._interim_task:
            await self.cancel_task(self._interim_task)
            self._interim_task = None

    async def _interim_task_handler(self):
        """Periodically transcribe the buffered audio with the interim model."""
        while True:
            started = time.monotonic()

            # Snapshot the buffer: it keeps growing while the interim model
            # reads the audio in a worker thread.
            audio = bytes(self._audio_buffer)
            if len(audio) >= self._interim_min_bytes:
                try:
                    text = await self.run_interim_stt(audio)
                except Exception as e:
                    text = None
                    logger.warning(f"{self} interim transcription failed: {e}")

                # Consecutive passes often return identical text (e.g. during
                # a pause); only push changed results.
                if text and text != self._last_interim_text:
                    self._last_interim_text = text
                    language = assert_given(self._settings.language)
                    await self._handle_transcription(text, False, language)
                    await self.push_frame(
                        InterimTranscriptionFrame(text, self._user_id, time_now_iso8601(), language)
                    )

            # Aim for one pass per interval; if inference took longer than the
            # interval, yield briefly instead of busy-looping.
            interval = assert_given(self._settings.interim_interval)
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.05, interval - elapsed))

    def _load(self):
        """Loads the final and interim Whisper models.

        Note:
            If this is the first time these models are being run,
            they will take time to download from the Hugging Face model hub.
        """
        logger.debug("Loading Whisper models...")
        model = assert_given(self._settings.model)
        interim_model = assert_given(self._settings.interim_model)
        if model is None:
            raise ValueError("Whisper model must be specified")
        # Separate instances (even for identical model names) so an in-flight
        # interim pass never contends with the final pass.
        self._model = WhisperModel(model, device=self._device, compute_type=self._compute_type)
        self._interim_model = WhisperModel(
            interim_model, device=self._device, compute_type=self._compute_type
        )
        logger.debug("Loaded Whisper models")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _transcribe(self, model: WhisperModel, audio: bytes) -> str:
        """Run one Whisper pass over raw 16-bit PCM audio.

        Args:
            model: The Whisper model to run.
            audio: Raw audio bytes in 16-bit signed PCM format.

        Returns:
            The transcribed text with non-speech segments filtered out, or an
            empty string if nothing was transcribed.
        """
        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        language = assert_given(self._settings.language)

        def transcribe() -> list:
            # transcribe() returns a lazy generator; consume it here so all
            # decoding happens in this worker thread, not on the event loop.
            segments, _ = model.transcribe(audio_float, language=language)
            return list(segments)

        segments = await asyncio.to_thread(transcribe)

        text = ""
        no_speech_prob_threshold = assert_given(self._settings.no_speech_prob)
        for segment in segments:
            if (
                no_speech_prob_threshold is not None
                and segment.no_speech_prob < no_speech_prob_threshold
            ):
                text += f"{segment.text} "
        return text.strip()

    async def run_interim_stt(self, audio: bytes) -> str | None:
        """Transcribe the audio buffered so far with the interim model.

        Args:
            audio: Raw audio bytes in 16-bit PCM format.

        Returns:
            The partial transcription, or None if nothing was transcribed.
        """
        if not self._interim_model:
            return None
        return await self._transcribe(self._interim_model, audio) or None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a complete speech segment with the final model.

        Called with the full buffered segment when the user stops speaking.

        Args:
            audio: Raw audio bytes in 16-bit PCM format.

        Yields:
            Frame: A finalized TranscriptionFrame containing the transcribed
                text, or an ErrorFrame if transcription fails.
        """
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()
        text = await self._transcribe(self._model, audio)
        await self.stop_processing_metrics()

        if text:
            language = assert_given(self._settings.language)
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                finalized=True,
            )
