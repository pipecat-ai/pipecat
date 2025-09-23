#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ElevenLabs speech-to-text service implementation.

This module provides integration with ElevenLabs' Speech-to-Text API for transcription
using segmented audio processing. The service uploads audio files and receives
transcription results directly.
"""

import io
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


def language_to_elevenlabs_language(language: Language) -> Optional[str]:
    """Convert a Language enum to ElevenLabs language code.

    Source:
        https://elevenlabs.io/docs/capabilities/speech-to-text

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding ElevenLabs language code, or None if not supported.
    """
    BASE_LANGUAGES = {
        Language.AF: "afr",  # Afrikaans
        Language.AM: "amh",  # Amharic
        Language.AR: "ara",  # Arabic
        Language.HY: "hye",  # Armenian
        Language.AS: "asm",  # Assamese
        Language.AST: "ast",  # Asturian
        Language.AZ: "aze",  # Azerbaijani
        Language.BE: "bel",  # Belarusian
        Language.BN: "ben",  # Bengali
        Language.BS: "bos",  # Bosnian
        Language.BG: "bul",  # Bulgarian
        Language.MY: "mya",  # Burmese
        Language.YUE: "yue",  # Cantonese
        Language.CA: "cat",  # Catalan
        Language.CEB: "ceb",  # Cebuano
        Language.NY: "nya",  # Chichewa
        Language.HR: "hrv",  # Croatian
        Language.CS: "ces",  # Czech
        Language.DA: "dan",  # Danish
        Language.NL: "nld",  # Dutch
        Language.EN: "eng",  # English
        Language.ET: "est",  # Estonian
        Language.FIL: "fil",  # Filipino
        Language.FI: "fin",  # Finnish
        Language.FR: "fra",  # French
        Language.FF: "ful",  # Fulah
        Language.GL: "glg",  # Galician
        Language.LG: "lug",  # Ganda
        Language.KA: "kat",  # Georgian
        Language.DE: "deu",  # German
        Language.EL: "ell",  # Greek
        Language.GU: "guj",  # Gujarati
        Language.HA: "hau",  # Hausa
        Language.HE: "heb",  # Hebrew
        Language.HI: "hin",  # Hindi
        Language.HU: "hun",  # Hungarian
        Language.IS: "isl",  # Icelandic
        Language.IG: "ibo",  # Igbo
        Language.ID: "ind",  # Indonesian
        Language.GA: "gle",  # Irish
        Language.IT: "ita",  # Italian
        Language.JA: "jpn",  # Japanese
        Language.JV: "jav",  # Javanese
        Language.KEA: "kea",  # Kabuverdianu
        Language.KN: "kan",  # Kannada
        Language.KK: "kaz",  # Kazakh
        Language.KM: "khm",  # Khmer
        Language.KO: "kor",  # Korean
        Language.KU: "kur",  # Kurdish
        Language.KY: "kir",  # Kyrgyz
        Language.LO: "lao",  # Lao
        Language.LV: "lav",  # Latvian
        Language.LN: "lin",  # Lingala
        Language.LT: "lit",  # Lithuanian
        Language.LUO: "luo",  # Luo
        Language.LB: "ltz",  # Luxembourgish
        Language.MK: "mkd",  # Macedonian
        Language.MS: "msa",  # Malay
        Language.ML: "mal",  # Malayalam
        Language.MT: "mlt",  # Maltese
        Language.ZH: "zho",  # Mandarin Chinese
        Language.MI: "mri",  # Māori
        Language.MR: "mar",  # Marathi
        Language.MN: "mon",  # Mongolian
        Language.NE: "nep",  # Nepali
        Language.NSO: "nso",  # Northern Sotho
        Language.NO: "nor",  # Norwegian
        Language.OC: "oci",  # Occitan
        Language.OR: "ori",  # Odia
        Language.PS: "pus",  # Pashto
        Language.FA: "fas",  # Persian
        Language.PL: "pol",  # Polish
        Language.PT: "por",  # Portuguese
        Language.PA: "pan",  # Punjabi
        Language.RO: "ron",  # Romanian
        Language.RU: "rus",  # Russian
        Language.SR: "srp",  # Serbian
        Language.SN: "sna",  # Shona
        Language.SD: "snd",  # Sindhi
        Language.SK: "slk",  # Slovak
        Language.SL: "slv",  # Slovenian
        Language.SO: "som",  # Somali
        Language.ES: "spa",  # Spanish
        Language.SW: "swa",  # Swahili
        Language.SV: "swe",  # Swedish
        Language.TA: "tam",  # Tamil
        Language.TG: "tgk",  # Tajik
        Language.TE: "tel",  # Telugu
        Language.TH: "tha",  # Thai
        Language.TR: "tur",  # Turkish
        Language.UK: "ukr",  # Ukrainian
        Language.UMB: "umb",  # Umbundu
        Language.UR: "urd",  # Urdu
        Language.UZ: "uzb",  # Uzbek
        Language.VI: "vie",  # Vietnamese
        Language.CY: "cym",  # Welsh
        Language.WO: "wol",  # Wolof
        Language.XH: "xho",  # Xhosa
        Language.ZU: "zul",  # Zulu
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class ElevenLabsSTTService(SegmentedSTTService):
    """Speech-to-text service using ElevenLabs' file-based API.

    This service uses ElevenLabs' Speech-to-Text API to perform transcription on audio
    segments. It inherits from SegmentedSTTService to handle audio buffering and speech detection.
    The service uploads audio files to ElevenLabs and receives transcription results directly.
    """

    class InputParams(BaseModel):
        """Configuration parameters for ElevenLabs STT API.

        Parameters:
            language: Target language for transcription.
            tag_audio_events: Whether to include audio events like (laughter), (coughing), in the transcription.
        """

        language: Optional[Language] = None
        tag_audio_events: bool = True

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.elevenlabs.io",
        model: str = "scribe_v1",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the ElevenLabs STT service.

        Args:
            api_key: ElevenLabs API key for authentication.
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            base_url: Base URL for ElevenLabs API.
            model: Model ID for transcription. Defaults to "scribe_v1".
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            params: Configuration parameters for the STT service.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or ElevenLabsSTTService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._model_id = model
        self._tag_audio_events = params.tag_audio_events

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "eng",
        }

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, as ElevenLabs STT service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to ElevenLabs service-specific language code.

        Args:
            language: The language to convert.

        Returns:
            The ElevenLabs-specific language code, or None if not supported.
        """
        return language_to_elevenlabs_language(language)

    async def set_language(self, language: Language):
        """Set the transcription language.

        Args:
            language: The language to use for speech-to-text transcription.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = self.language_to_service_language(language)

    async def set_model(self, model: str):
        """Set the STT model.

        Args:
            model: The model name to use for transcription.

        Note:
            ElevenLabs STT API does not currently support model selection.
            This method is provided for interface compatibility.
        """
        await super().set_model(model)
        logger.info(f"Model setting [{model}] noted, but ElevenLabs STT uses default model")

    async def _transcribe_audio(self, audio_data: bytes) -> dict:
        """Upload audio data to ElevenLabs and get transcription result.

        Args:
            audio_data: Raw audio bytes in WAV format.

        Returns:
            The transcription result data.

        Raises:
            Exception: If transcription fails or returns an error.
        """
        url = f"{self._base_url}/v1/speech-to-text"
        headers = {"xi-api-key": self._api_key}

        # Create form data with the audio file
        data = aiohttp.FormData()
        data.add_field(
            "file",
            io.BytesIO(audio_data),
            filename="audio.wav",
            content_type="audio/x-wav",
        )

        # Add required model_id, language_code, and tag_audio_events
        data.add_field("model_id", self._model_id)
        data.add_field("language_code", self._settings["language"])
        data.add_field("tag_audio_events", str(self._tag_audio_events).lower())

        async with self._session.post(url, data=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"ElevenLabs transcription error: {error_text}")
                raise Exception(f"Transcription failed with status {response.status}: {error_text}")

            result = await response.json()
            return result

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe an audio segment using ElevenLabs' STT API.

        Args:
            audio: Raw audio bytes in WAV format (already converted by base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text, or ErrorFrame on failure.

        Note:
            The audio is already in WAV format from the SegmentedSTTService.
            Only non-empty transcriptions are yielded.
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            # Upload audio and get transcription result directly
            result = await self._transcribe_audio(audio)

            # Extract transcription text
            text = result.get("text", "").strip()
            if text:
                # Use the language_code returned by the API
                detected_language = result.get("language_code", "eng")

                await self._handle_transcription(text, True, detected_language)
                logger.debug(f"Transcription: [{text}]")

                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    detected_language,
                    result=result,
                )

        except Exception as e:
            logger.error(f"ElevenLabs STT error: {e}")
            yield ErrorFrame(f"ElevenLabs STT error: {str(e)}")
