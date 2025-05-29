#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import fal_client
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Fal, you need to `pip install pipecat-ai[fal]`. Also, set `FAL_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def language_to_fal_language(language: Language) -> Optional[str]:
    """Language support for Fal's Wizper API."""
    BASE_LANGUAGES = {
        Language.AF: "af",
        Language.AM: "am",
        Language.AR: "ar",
        Language.AS: "as",
        Language.AZ: "az",
        Language.BA: "ba",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BO: "bo",
        Language.BR: "br",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FO: "fo",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HA: "ha",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HT: "ht",
        Language.HU: "hu",
        Language.HY: "hy",
        Language.ID: "id",
        Language.IS: "is",
        Language.IT: "it",
        Language.JA: "ja",
        Language.JW: "jw",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KM: "km",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LA: "la",
        Language.LB: "lb",
        Language.LN: "ln",
        Language.LO: "lo",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MG: "mg",
        Language.MI: "mi",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MN: "mn",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.MY: "my",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NN: "nn",
        Language.NO: "no",
        Language.OC: "oc",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SA: "sa",
        Language.SD: "sd",
        Language.SI: "si",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SN: "sn",
        Language.SO: "so",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SU: "su",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TG: "tg",
        Language.TH: "th",
        Language.TK: "tk",
        Language.TL: "tl",
        Language.TR: "tr",
        Language.TT: "tt",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.UZ: "uz",
        Language.VI: "vi",
        Language.YI: "yi",
        Language.YO: "yo",
        Language.ZH: "zh",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class FalSTTService(SegmentedSTTService):
    """Speech-to-text service using Fal's Wizper API.

    This service uses Fal's Wizper API to perform speech-to-text transcription on audio
    segments. It inherits from SegmentedSTTService to handle audio buffering and speech detection.

    Args:
        api_key: Fal API key. If not provided, will check FAL_KEY environment variable.
        sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
        params: Configuration parameters for the Wizper API.
        **kwargs: Additional arguments passed to SegmentedSTTService.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Fal's Wizper API.

        Attributes:
            language: Language of the audio input. Defaults to English.
            task: Task to perform ('transcribe' or 'translate'). Defaults to 'transcribe'.
            chunk_level: Level of chunking ('segment'). Defaults to 'segment'.
            version: Version of Wizper model to use. Defaults to '3'.
        """

        language: Optional[Language] = Language.EN
        task: str = "transcribe"
        chunk_level: str = "segment"
        version: str = "3"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or FalSTTService.InputParams()

        if api_key:
            os.environ["FAL_KEY"] = api_key
        elif "FAL_KEY" not in os.environ:
            raise ValueError(
                "FAL_KEY must be provided either through api_key parameter or environment variable"
            )

        self._fal_client = fal_client.AsyncClient(key=api_key or os.getenv("FAL_KEY"))
        self._settings = {
            "task": params.task,
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en",
            "chunk_level": params.chunk_level,
            "version": params.version,
        }

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_fal_language(language)

    async def set_language(self, language: Language):
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = self.language_to_service_language(language)

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes an audio segment using Fal's Wizper API.

        Args:
            audio: Raw audio bytes in WAV format (already converted by base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text.

        Note:
            The audio is already in WAV format from the SegmentedSTTService.
            Only non-empty transcriptions are yielded.
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            # Send to Fal directly (audio is already in WAV format from base class)
            data_uri = fal_client.encode(audio, "audio/x-wav")
            response = await self._fal_client.run(
                "fal-ai/wizper",
                arguments={"audio_url": data_uri, **self._settings},
            )

            if response and "text" in response:
                text = response["text"].strip()
                if text:  # Only yield non-empty text
                    await self._handle_transcription(text, True, self._settings["language"])
                    logger.debug(f"Transcription: [{text}]")
                    yield TranscriptionFrame(
                        text,
                        "",
                        time_now_iso8601(),
                        Language(self._settings["language"]),
                        result=response,
                    )

        except Exception as e:
            logger.error(f"Fal Wizper error: {e}")
            yield ErrorFrame(f"Fal Wizper error: {str(e)}")
