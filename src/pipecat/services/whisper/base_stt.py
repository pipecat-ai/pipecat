#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base class for Whisper-based speech-to-text services.

This module provides common functionality for services implementing the Whisper API
interface, including language mapping, metrics generation, and error handling.
"""

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from loguru import logger
from openai import AsyncOpenAI
from openai.types.audio import Transcription

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import WHISPER_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


@dataclass
class BaseWhisperSTTSettings(STTSettings):
    """Settings for Whisper API-based STT services.

    Parameters:
        base_url: API base URL.
        prompt: Optional text to guide the model's style or continue
            a previous segment.
        temperature: Sampling temperature between 0 and 1.
    """

    base_url: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prompt: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


def language_to_whisper_language(language: Language) -> Optional[str]:
    """Maps pipecat Language enum to Whisper API language codes.

    Language support for Whisper API.
    Docs: https://platform.openai.com/docs/guides/speech-to-text#supported-languages

    Args:
        language: A Language enum value representing the input language.

    Returns:
        str or None: The corresponding Whisper language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.AF: "af",
        Language.AR: "ar",
        Language.HY: "hy",
        Language.AZ: "az",
        Language.BE: "be",
        Language.BS: "bs",
        Language.BG: "bg",
        Language.CA: "ca",
        Language.ZH: "zh",
        Language.HR: "hr",
        Language.CS: "cs",
        Language.DA: "da",
        Language.NL: "nl",
        Language.EN: "en",
        Language.ET: "et",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.DE: "de",
        Language.EL: "el",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HU: "hu",
        Language.IS: "is",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KN: "kn",
        Language.KK: "kk",
        Language.KO: "ko",
        Language.LV: "lv",
        Language.LT: "lt",
        Language.MK: "mk",
        Language.MS: "ms",
        Language.MR: "mr",
        Language.MI: "mi",
        Language.NE: "ne",
        Language.NO: "no",
        Language.FA: "fa",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SR: "sr",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.ES: "es",
        Language.SW: "sw",
        Language.SV: "sv",
        Language.TL: "tl",
        Language.TA: "ta",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.CY: "cy",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


class BaseWhisperSTTService(SegmentedSTTService):
    """Base class for Whisper-based speech-to-text services.

    Provides common functionality for services implementing the Whisper API interface,
    including metrics generation and error handling.
    """

    _settings: BaseWhisperSTTSettings

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[Language] = Language.EN,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        include_prob_metrics: bool = False,
        ttfs_p99_latency: Optional[float] = WHISPER_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Whisper STT service.

        Args:
            model: Name of the Whisper model to use.
            api_key: Service API key. Defaults to None.
            base_url: Service API base URL. Defaults to None.
            language: Language of the audio input. Defaults to English.
            prompt: Optional text to guide the model's style or continue a previous segment.
            temperature: Sampling temperature between 0 and 1. Defaults to 0.0.
            include_prob_metrics: If True, enables probability metrics in API response.
                Each service implements this differently (see child classes).
                Defaults to False.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        super().__init__(
            ttfs_p99_latency=ttfs_p99_latency,
            settings=BaseWhisperSTTSettings(
                model=model,
                language=self.language_to_service_language(language or Language.EN),
                base_url=base_url,
                prompt=prompt,
                temperature=temperature,
            ),
            **kwargs,
        )
        self._client = self._create_client(api_key, base_url)
        self._language = self._settings.language
        self._prompt = prompt
        self._temperature = temperature
        self._include_prob_metrics = include_prob_metrics

    def _create_client(self, api_key: Optional[str], base_url: Optional[str]):
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, syncing instance variables.

        Keeps ``_language``, ``_prompt``, and ``_temperature`` in sync with
        the settings fields.
        """
        changed = await super()._update_settings(delta)

        if "language" in changed:
            self._language = self._settings.language
        if "prompt" in changed:
            self._prompt = self._settings.prompt
        if "temperature" in changed:
            self._temperature = self._settings.temperature

        return changed

    def can_generate_metrics(self) -> bool:
        """Whether this service can generate processing metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert from pipecat Language to service language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            str or None: The corresponding service language code, or None if not supported.
        """
        return language_to_whisper_language(language)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio data to text.

        Args:
            audio: Raw audio data to transcribe.

        Yields:
            Frame: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.
        """
        try:
            await self.start_processing_metrics()

            response = await self._transcribe(audio)

            await self.stop_processing_metrics()

            text = response.text.strip()

            if text:
                await self._handle_transcription(text, True, self._language)
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    result=response,
                )
            else:
                logger.warning("Received empty transcription from API")

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def _transcribe(self, audio: bytes) -> Transcription:
        """Transcribe audio data to text.

        Args:
            audio: Raw audio data in WAV format.

        Returns:
            Transcription: Object containing the transcribed text.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
