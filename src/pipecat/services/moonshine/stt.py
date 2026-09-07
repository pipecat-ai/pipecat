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
from typing import Any

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import assert_given

try:
    from moonshine_voice import (
        Transcriber,
        get_model_for_language,
        model_arch_to_string,
        string_to_model_arch,
        supported_languages,
        supported_languages_friendly,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Moonshine, you need to `uv add "pipecat-ai[moonshine]"`.')
    raise ImportError(f"Missing module: {e}") from e

# Moonshine expects 16 kHz mono PCM audio.
MOONSHINE_SAMPLE_RATE = 16000


def language_to_moonshine_language(language: Language) -> str:
    """Convert a pipecat Language to a Moonshine language code.

    Moonshine publishes one model per language, keyed by base ISO code, so
    regional variants (``Language.ES_MX``) resolve to their base code (``"es"``).

    Args:
        language: The Language enum value to convert.

    Returns:
        The Moonshine language code.
    """
    LANGUAGE_MAP = {
        Language.AR: "ar",
        Language.EN: "en",
        Language.ES: "es",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.UK: "uk",
        Language.VI: "vi",
        Language.ZH: "zh",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def moonshine_language_to_frame_language(language: str | None) -> Language | None:
    """Map a Moonshine language code back to a pipecat ``Language`` when possible."""
    if language is None:
        return None
    try:
        return Language(language)
    except ValueError:
        return None


class Model(StrEnum):
    """Well-known Moonshine model architectures.

    Pass a member (or the equivalent string) as ``MoonshineSTTService.Settings``'s
    ``model``. The larger models (``SMALL_STREAMING``, ``MEDIUM_STREAMING``) ship
    only in streaming form, but transcribe a whole segment in batch just the same.

    Which architectures exist depends on the language: the streaming ones are
    published for English only, and most other languages ship a single model. An
    architecture unavailable for the configured language falls back to the best
    one published for it.

    Parameters:
        TINY: Smallest and fastest, lowest accuracy.
        BASE: Good size/accuracy balance.
        TINY_STREAMING: Streaming-capable ``tiny``.
        BASE_STREAMING: Streaming-capable ``base``.
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
        language: Language for transcription. Moonshine publishes models for
            Arabic, Chinese, English, Japanese, Korean, Spanish, Ukrainian, and
            Vietnamese; regional variants resolve to their base code.
    """


class MoonshineSTTService(SegmentedSTTService):
    """Transcribe audio with a locally-downloaded Moonshine ONNX model.

    Runs on the CPU via ONNX Runtime, so it needs no GPU and no API key. The model
    downloads once on first use and is cached. Each VAD-segmented utterance is
    transcribed in a single batch call (``transcribe_without_streaming``); any
    model works, including the streaming-capable ones. Audio is expected as 16-bit
    mono PCM at 16 kHz.

    Models are language-specific, so a language change reloads the model. The
    non-English models are released under the non-commercial Moonshine Community
    License (https://www.moonshine.ai/license).
    """

    Settings = MoonshineSTTSettings
    _settings: Settings

    @property
    def wants_wav_segments(self) -> bool:
        """Receive segments as raw 16-bit PCM, which the model reads directly."""
        return False

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

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a pipecat language into a Moonshine language code."""
        return language_to_moonshine_language(language)

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, reloading the model when it changes.

        Moonshine models are per-language and per-architecture, so a new
        ``language`` or ``model`` only takes effect once the model is reloaded.
        """
        changed = await super()._update_settings(delta)

        if "language" in changed or "model" in changed:
            try:
                self._transcriber = await asyncio.to_thread(self._load)
            except Exception as e:
                # Keep transcribing with the model already loaded.
                logger.error(f"{self} error loading Moonshine model: {e}")
                await self.push_error(f"Moonshine model load error: {e}", e)

        return changed

    def _load(self) -> Transcriber:
        """Download (first time) and load the Moonshine model.

        Note:
            The first run downloads the model from the Moonshine model hub; later
            runs load it from the local cache.

        Raises:
            ValueError: If no language is set, or Moonshine publishes no model for it.
        """
        logger.debug("Loading Moonshine model...")
        model = assert_given(self._settings.model)
        model_str = model.value if isinstance(model, Model) else str(model)

        language = assert_given(self._settings.language)
        if language is None:
            raise ValueError("Moonshine requires a language; its models are language-specific")
        lang_code = str(language)
        if lang_code not in supported_languages():
            raise ValueError(
                f"Moonshine does not support language '{lang_code}'. "
                f"Supported languages: {supported_languages_friendly()}"
            )

        wanted_arch = string_to_model_arch(model_str)
        try:
            model_path, model_arch = get_model_for_language(lang_code, wanted_arch)
        except ValueError:
            # The architecture isn't published for this language; take its best model.
            model_path, model_arch = get_model_for_language(lang_code)
            arch_str = model_arch_to_string(model_arch)
            logger.warning(
                f"Moonshine model '{model_str}' is unavailable for '{lang_code}'; "
                f"using '{arch_str}' instead"
            )
            self._settings.model = arch_str

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

        lang_code = assert_given(self._settings.language)
        language = moonshine_language_to_frame_language(str(lang_code) if lang_code else None)
        if text:
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)
