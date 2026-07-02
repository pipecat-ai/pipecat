#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FunASR speech-to-text service with locally-downloaded models.

This module implements speech-to-text using locally-run FunASR models such as
SenseVoice. SenseVoice is a multilingual model (Chinese, Cantonese, English,
Japanese, Korean and more) with leading Chinese accuracy and non-autoregressive
(fast) inference, making it a strong fully-local STT option for voice agents.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from typing_extensions import override

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, assert_given
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use FunASR, you need to `pip install "pipecat-ai[funasr]"`.')
    raise ImportError(f"Missing module: {e}") from e


# Language codes natively supported by SenseVoice; anything else falls back to
# automatic language detection.
_FUNASR_LANGUAGES = {"zh", "en", "ja", "ko", "yue", "nospeech"}


def language_to_funasr_language(language: Language | str | None) -> str:
    """Map a language value to a SenseVoice language code.

    Args:
        language: A pipecat language, raw language code, or ``None`` for
            auto-detection.

    Returns:
        A SenseVoice language code (e.g. ``"zh"``), or ``"auto"``.
    """
    if language is None:
        return "auto"
    if isinstance(language, Language):
        code = str(language.value).split("-")[0].lower()
    else:
        code = str(language).split("-")[0].lower()
    return code if code in _FUNASR_LANGUAGES else "auto"


def funasr_language_to_frame_language(language: str | None) -> Language | None:
    """Map a FunASR language code back to a pipecat ``Language`` when possible."""
    if language is None or language == "auto":
        return None
    try:
        return Language(language)
    except ValueError:
        return None


@dataclass
class FunASRSTTSettings(STTSettings):
    """Settings for ``FunASRSTTService``.

    ``model`` and ``language`` are inherited from ``STTSettings``.

    Parameters:
        use_itn: Apply inverse text normalization (e.g. "nine" -> "9").
    """

    use_itn: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class FunASRSTTService(SegmentedSTTService):
    """Speech-to-text using a locally-downloaded FunASR model (e.g. SenseVoice).

    Transcribes VAD-segmented speech with a local FunASR model. SenseVoice is
    multilingual (Chinese, Cantonese, English, Japanese, Korean, ...) with strong
    Chinese accuracy and fast non-autoregressive inference.

    Audio is expected as 16 kHz, 16-bit signed PCM (the pipecat STT default).
    """

    Settings = FunASRSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        device: str = "cpu",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the FunASR STT service.

        Args:
            device: Inference device, ``"cpu"`` or ``"cuda"``.
            settings: Runtime-updatable settings (``model``, ``language``, ``use_itn``).
            **kwargs: Additional arguments passed to ``SegmentedSTTService``.
        """
        default_settings = self.Settings(
            model="iic/SenseVoiceSmall",
            language=Language.EN,
            use_itn=True,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)
        self._device = device

        model = assert_given(self._settings.model)
        if model is None:
            raise ValueError("FunASR model must be specified")

        logger.debug(f"Loading FunASR model {model}...")
        self._model = AutoModel(model=model, device=device, disable_update=True)
        logger.debug("Loaded FunASR model")

    @override
    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a pipecat language into a FunASR language code."""
        return language_to_funasr_language(language)

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate processing metrics."""
        return True

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    @override
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe an audio segment with FunASR.

        Args:
            audio: Raw audio bytes in 16-bit signed PCM format (16 kHz mono).

        Yields:
            Frame: A ``TranscriptionFrame`` with the recognized text, or an
                ``ErrorFrame`` if transcription fails.
        """
        if not self._model:
            yield ErrorFrame("FunASR model not available")
            return

        await self.start_processing_metrics()

        # 16-bit signed PCM -> float32 in [-1, 1].
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        language = language_to_funasr_language(assert_given(self._settings.language))
        frame_language = funasr_language_to_frame_language(language)
        use_itn = assert_given(self._settings.use_itn)

        def _transcribe() -> str:
            result = self._model.generate(
                input=audio_float,
                cache={},
                language=language,
                use_itn=use_itn,
            )
            return rich_transcription_postprocess(result[0]["text"]).strip()

        try:
            text = await asyncio.to_thread(_transcribe)
        except Exception as e:
            logger.error(f"{self} error running FunASR: {e}")
            await self.stop_processing_metrics()
            yield ErrorFrame(f"FunASR transcription error: {e}")
            return

        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription: [{text}]")
            await self._handle_transcription(text, True, frame_language)
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                frame_language,
            )
