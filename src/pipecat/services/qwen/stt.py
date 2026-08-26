#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Qwen3-ASR speech-to-text service with locally-downloaded models.

This module implements speech-to-text using the locally-run Qwen3-ASR model
family (Qwen/Qwen3-ASR-0.6B, 1.7B, 8B). Qwen3-ASR is a multilingual,
autoregressive ASR model that runs on CUDA via the ``qwen-asr`` package.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from loguru import logger
from typing_extensions import override

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import torch
    from qwen_asr import Qwen3ASRModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Qwen3-ASR, you need to `uv add "pipecat-ai[qwen-asr]"`.')
    raise ImportError(f"Missing module: {e}") from e

# Qwen3-ASR natively operates on 16 kHz mono PCM.
_SAMPLE_RATE = 16000

_LANGUAGE_MAP = {
    Language.EN: "English",
    Language.ZH: "Chinese",
    Language.ES: "Spanish",
    Language.FR: "French",
    Language.DE: "German",
    Language.JA: "Japanese",
    Language.KO: "Korean",
    Language.PT: "Portuguese",
    Language.RU: "Russian",
    Language.AR: "Arabic",
    Language.IT: "Italian",
    Language.HI: "Hindi",
}


class Model(StrEnum):
    """Qwen3-ASR model size options.

    Parameters:
        ASR_0_6B: Smallest model, fastest inference, lower accuracy.
        ASR_1_7B: Best accuracy/speed balance. Recommended default.
        ASR_8B: Highest accuracy, slower inference, more VRAM required.
    """

    ASR_0_6B = "Qwen/Qwen3-ASR-0.6B"
    ASR_1_7B = "Qwen/Qwen3-ASR-1.7B"
    ASR_8B   = "Qwen/Qwen3-ASR-8B"


@dataclass
class Qwen3STTSettings(STTSettings):
    """Settings for ``Qwen3STTService``.

    ``model`` and ``language`` are inherited from ``STTSettings``.

    Parameters:
        max_new_tokens: Maximum tokens the model may generate per utterance.
            Higher values allow longer transcriptions but increase latency.
    """

    max_new_tokens: int = 256


class Qwen3STTService(SegmentedSTTService):
    """Speech-to-text using a locally-downloaded Qwen3-ASR model.

    Transcribes VAD-segmented speech with a local Qwen3-ASR model.
    The model is loaded once at construction and reused for each transcription.

    Audio is expected as 16 kHz, 16-bit signed PCM (the pipecat STT default).
    Requires a CUDA-capable GPU.

    For deployments with many concurrent sessions, note that each service
    instance loads its own copy of the model into VRAM. To share one model
    across sessions, subclass this service and override ``_load()`` to inject
    a pre-loaded ``Qwen3ASRModel``.

    Example::

        stt = Qwen3STTService(
            model=Model.ASR_1_7B,
            device="cuda:0",
            language=Language.EN,
        )
    """

    Settings = Qwen3STTSettings
    _settings: Settings

    @property
    def wants_wav_segments(self) -> bool:
        """Receive segments as raw 16-bit PCM — no WAV wrapping needed."""
        return False

    def __init__(
        self,
        *,
        model: str | Model = Model.ASR_1_7B,
        device: str = "cuda:0",
        language: Language | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Qwen3-ASR STT service.

        Args:
            model: The Qwen3-ASR model to use. Can be a :class:`Model` enum
                value or a HuggingFace model ID string.
                Defaults to :attr:`Model.ASR_1_7B`.
            device: The CUDA device to load the model onto. Defaults to
                ``"cuda:0"``.
            language: The default transcription language. Defaults to
                :attr:`Language.EN` when not provided via ``settings``.
            settings: Runtime-updatable settings (``model``, ``language``,
                ``max_new_tokens``). Values in ``settings`` take precedence
                over the ``language`` argument.
            **kwargs: Additional arguments passed to ``SegmentedSTTService``.
        """
        default_settings = self.Settings(
            model=model if isinstance(model, str) else model.value,
            language=language or Language.EN,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)
        self._device = device
        self._qwen_model: "Qwen3ASRModel | None" = None
        self._load()

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Map a pipecat ``Language`` to the Qwen3-ASR language string.

        Args:
            language: The pipecat language to convert.

        Returns:
            The Qwen3-ASR language string (e.g. ``"English"``), or ``None``
            if the language is not in the supported set.
        """
        return _LANGUAGE_MAP.get(language)

    def _load(self) -> None:
        """Load the Qwen3-ASR model from HuggingFace (cached after first run)."""
        logger.debug(f"Loading Qwen3-ASR model: {self._settings.model} ...")
        self._qwen_model = Qwen3ASRModel.from_pretrained(
            self._settings.model,
            dtype=torch.bfloat16,
            device_map=self._device,
            max_new_tokens=self._settings.max_new_tokens,
        )
        logger.debug("Qwen3-ASR model loaded")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    @override
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe an audio segment with Qwen3-ASR.

        Args:
            audio: Raw audio bytes in 16-bit signed PCM format (16 kHz mono).

        Yields:
            Frame: A ``TranscriptionFrame`` with the recognized text, or an
                ``ErrorFrame`` if the model is unavailable or inference fails.
        """
        if not self._qwen_model:
            yield ErrorFrame("Qwen3-ASR model not available")
            return

        await self.start_processing_metrics()

        # 16-bit signed PCM -> float32 in [-1, 1].
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        language = self._settings.language
        lang_str = self.language_to_service_language(language) if language else "English"

        try:
            results = await asyncio.to_thread(
                self._qwen_model.transcribe,
                (audio_float, _SAMPLE_RATE),
                language=lang_str,
            )
            text = results[0].text.strip() if results else ""
        except Exception as e:
            logger.exception("Qwen3-ASR transcription failed")
            await self.stop_processing_metrics()
            yield ErrorFrame(error=str(e))
            return

        await self.stop_processing_metrics()

        if text:
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)
