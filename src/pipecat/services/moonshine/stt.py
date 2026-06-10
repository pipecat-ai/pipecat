#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Moonshine speech-to-text service with locally-downloaded ONNX models.

`Moonshine <https://github.com/moonshine-ai/moonshine>`_ is a small, fast ASR
family that runs on the CPU via ONNX Runtime -- no GPU and no API key. This module
transcribes audio segments with a locally-downloaded Moonshine model (downloaded
once on first use and cached).
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings, assert_given
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from moonshine_voice import Transcriber, get_model_for_language, string_to_model_arch
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Moonshine, you need to `uv add "pipecat-ai[moonshine]"`.')
    raise ImportError(f"Missing module: {e}") from e

# Moonshine expects 16 kHz mono PCM audio.
MOONSHINE_SAMPLE_RATE = 16000


class Model(StrEnum):
    """Well-known Moonshine model architectures.

    Pass a member (or the equivalent string) as ``MoonshineSTTService.Settings``'s
    ``model``. The larger models (``SMALL_STREAMING``, ``MEDIUM_STREAMING``) ship
    only in streaming form, but transcribe a whole segment in batch just the same.

    Parameters:
        TINY: Smallest and fastest, lowest accuracy.
        BASE: Good size/accuracy balance.
        TINY_STREAMING: Streaming-capable ``tiny``.
        BASE_STREAMING: Streaming-capable ``base`` (not available for every language).
        SMALL_STREAMING: Larger and more accurate than ``base`` (the default).
        MEDIUM_STREAMING: Largest, most accurate.
    """

    TINY = "tiny"
    BASE = "base"
    TINY_STREAMING = "tiny-streaming"
    BASE_STREAMING = "base-streaming"
    SMALL_STREAMING = "small-streaming"
    MEDIUM_STREAMING = "medium-streaming"


@dataclass
class MoonshineSTTSettings(STTSettings):
    """Settings for ``MoonshineSTTService``.

    Parameters:
        model: Moonshine model architecture, as a :class:`Model` or the equivalent
            string (e.g. ``Model.SMALL_STREAMING`` or ``"small-streaming"``).
            Defaults to ``Model.SMALL_STREAMING``.
        language: Language for transcription. Moonshine supports a handful of
            languages (English, Spanish, ...); the base code is used.
    """


class MoonshineSTTService(SegmentedSTTService):
    """Transcribe audio with a locally-downloaded Moonshine ONNX model.

    Runs on the CPU via ONNX Runtime, so it needs no GPU and no API key. The model
    downloads once on first use and is cached. Each VAD-segmented utterance is
    transcribed in a single batch call (``transcribe_without_streaming``); any
    model works, including the streaming-capable ones. Audio is expected as 16-bit
    mono PCM at 16 kHz.
    """

    Settings = MoonshineSTTSettings
    _settings: Settings

    def __init__(self, *, settings: Settings | None = None, **kwargs):
        """Initialize the Moonshine STT service.

        Args:
            settings: Runtime-updatable settings (``model``, ``language``).
            **kwargs: Additional arguments passed to ``SegmentedSTTService``.
        """
        default_settings = self.Settings(
            model=Model.SMALL_STREAMING.value,
            language=Language.EN,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(sample_rate=MOONSHINE_SAMPLE_RATE, settings=default_settings, **kwargs)

        self._transcriber = self._load()

    def can_generate_metrics(self) -> bool:
        """Indicate whether this service can generate metrics.

        Returns:
            True, as this service supports metric generation.
        """
        return True

    def _load(self) -> Transcriber:
        """Download (first time) and load the Moonshine model.

        Note:
            The first run downloads the model from the Moonshine model hub; later
            runs load it from the local cache.
        """
        logger.debug("Loading Moonshine model...")
        model = assert_given(self._settings.model)
        model_str = model.value if isinstance(model, Model) else str(model)
        language = str(assert_given(self._settings.language))
        lang_code = language.split("-")[0].lower()
        model_path, model_arch = get_model_for_language(lang_code, string_to_model_arch(model_str))
        transcriber = Transcriber(model_path, model_arch)
        logger.debug("Loaded Moonshine model")
        return transcriber

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio data using Moonshine.

        Args:
            audio: Raw 16-bit signed PCM mono audio at 16 kHz.

        Yields:
            Frame: A ``TranscriptionFrame`` with the transcribed text.
        """
        await self.start_processing_metrics()

        # Divide by 32768 because we have signed 16-bit data; Moonshine wants a list
        # of floats in [-1, 1].
        audio_float = (np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0).tolist()

        transcript = await asyncio.to_thread(
            self._transcriber.transcribe_without_streaming, audio_float, MOONSHINE_SAMPLE_RATE
        )
        text = " ".join(line.text for line in transcript.lines).strip()

        await self.stop_processing_metrics()

        language = self._settings.language
        language = language if isinstance(language, Language) else None
        if text:
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)
