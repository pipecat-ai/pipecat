#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator, Optional

from loguru import logger
from openai import AsyncOpenAI
from openai.types.audio import Transcription

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


def language_to_whisper_language(language: Language) -> Optional[str]:
    """Language support for Whisper API.

    Docs: https://platform.openai.com/docs/guides/speech-to-text#supported-languages
    """
    BASE_LANGUAGES = {
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

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class BaseWhisperSTTService(SegmentedSTTService):
    """Base class for Whisper-based speech-to-text services.

    Provides common functionality for services implementing the Whisper API interface,
    including metrics generation and error handling.

    Args:
        model: Name of the Whisper model to use.
        api_key: Service API key. Defaults to None.
        base_url: Service API base URL. Defaults to None.
        language: Language of the audio input. Defaults to English.
        prompt: Optional text to guide the model's style or continue a previous segment.
        temperature: Sampling temperature between 0 and 1. Defaults to 0.0.
        **kwargs: Additional arguments passed to SegmentedSTTService.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[Language] = Language.EN,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._client = self._create_client(api_key, base_url)
        self._language = self.language_to_service_language(language or Language.EN)
        self._prompt = prompt
        self._temperature = temperature

        self._settings = {
            "base_url": base_url,
            "language": self._language,
            "prompt": self._prompt,
            "temperature": self._temperature,
        }

    def _create_client(self, api_key: Optional[str], base_url: Optional[str]):
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def set_model(self, model: str):
        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_whisper_language(language)

    async def set_language(self, language: Language):
        """Set the language for transcription.

        Args:
            language: The Language enum value to use for transcription.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._language = language

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            response = await self._transcribe(audio)

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

            text = response.text.strip()

            if text:
                await self._handle_transcription(text, True, self._language)
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(text, "", time_now_iso8601())
            else:
                logger.warning("Received empty transcription from API")

        except Exception as e:
            logger.exception(f"Exception during transcription: {e}")
            yield ErrorFrame(f"Error during transcription: {str(e)}")

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
