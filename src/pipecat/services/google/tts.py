#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Cloud Text-to-Speech service implementations.

This module provides integration with Google Cloud Text-to-Speech API,
offering both HTTP-based synthesis with SSML support and streaming synthesis
for real-time applications.

It also includes GeminiTTSService which uses Gemini's TTS-specific models
for natural voice control and multi-speaker conversations.
"""

import json
import os
import warnings

from pipecat.utils.tracing.service_decorators import traced_tts

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from typing import Any, AsyncGenerator, List, Literal, Mapping, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language

try:
    from google.auth import default
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import texttospeech_v1
    from google.oauth2 import service_account

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_APPLICATION_CREDENTIALS` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def language_to_google_tts_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Google TTS language code.

    Source:
    https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Google TTS language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        # Arabic
        Language.AR: "ar-XA",
        # Bengali
        Language.BN: "bn-IN",
        Language.BN_IN: "bn-IN",
        # Bulgarian
        Language.BG: "bg-BG",
        Language.BG_BG: "bg-BG",
        # Croatian
        Language.HR: "hr-HR",
        Language.HR_HR: "hr-HR",
        # Czech
        Language.CS: "cs-CZ",
        Language.CS_CZ: "cs-CZ",
        # Danish
        Language.DA: "da-DK",
        Language.DA_DK: "da-DK",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_BE: "nl-BE",
        Language.NL_NL: "nl-NL",
        # English
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        # Estonian
        Language.ET: "et-EE",
        Language.ET_EE: "et-EE",
        # Finnish
        Language.FI: "fi-FI",
        Language.FI_FI: "fi-FI",
        # French
        Language.FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        Language.FR_FR: "fr-FR",
        # German
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        # Greek
        Language.EL: "el-GR",
        Language.EL_GR: "el-GR",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hebrew
        Language.HE: "he-IL",
        Language.HE_IL: "he-IL",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Hungarian
        Language.HU: "hu-HU",
        Language.HU_HU: "hu-HU",
        # Indonesian
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        # Italian
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Latvian
        Language.LV: "lv-LV",
        Language.LV_LV: "lv-LV",
        # Lithuanian
        Language.LT: "lt-LT",
        Language.LT_LT: "lt-LT",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Chinese (Mandarin)
        Language.ZH: "cmn-CN",
        Language.ZH_CN: "cmn-CN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Norwegian
        Language.NO: "nb-NO",
        Language.NB: "nb-NO",
        Language.NB_NO: "nb-NO",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese
        Language.PT: "pt-BR",
        Language.PT_BR: "pt-BR",
        # Romanian
        Language.RO: "ro-RO",
        Language.RO_RO: "ro-RO",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Serbian
        Language.SR: "sr-RS",
        Language.SR_RS: "sr-RS",
        # Slovak
        Language.SK: "sk-SK",
        Language.SK_SK: "sk-SK",
        # Slovenian
        Language.SL: "sl-SI",
        Language.SL_SI: "sl-SI",
        # Spanish
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
        # Swahili
        Language.SW: "sw-KE",
        Language.SW_KE: "sw-KE",
        # Swedish
        Language.SV: "sv-SE",
        Language.SV_SE: "sv-SE",
        # Tamil
        Language.TA: "ta-IN",
        Language.TA_IN: "ta-IN",
        # Telugu
        Language.TE: "te-IN",
        Language.TE_IN: "te-IN",
        # Thai
        Language.TH: "th-TH",
        Language.TH_TH: "th-TH",
        # Turkish
        Language.TR: "tr-TR",
        Language.TR_TR: "tr-TR",
        # Ukrainian
        Language.UK: "uk-UA",
        Language.UK_UA: "uk-UA",
        # Urdu
        Language.UR: "ur-IN",
        Language.UR_IN: "ur-IN",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


def language_to_gemini_tts_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Gemini TTS language code.

    Source:
    https://docs.cloud.google.com/text-to-speech/docs/gemini-tts#available_languages

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Gemini TTS language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        # Afrikaans (Preview)
        Language.AF: "af-ZA",
        Language.AF_ZA: "af-ZA",
        # Albanian (Preview)
        Language.SQ: "sq-AL",
        Language.SQ_AL: "sq-AL",
        # Amharic (Preview)
        Language.AM: "am-ET",
        Language.AM_ET: "am-ET",
        # Arabic
        Language.AR: "ar-EG",  # GA: Egypt
        Language.AR_EG: "ar-EG",
        Language.AR_001: "ar-001",  # Preview: World
        # Armenian (Preview)
        Language.HY: "hy-AM",
        Language.HY_AM: "hy-AM",
        # Azerbaijani (Preview)
        Language.AZ: "az-AZ",
        Language.AZ_AZ: "az-AZ",
        # Basque (Preview)
        Language.EU: "eu-ES",
        Language.EU_ES: "eu-ES",
        # Belarusian (Preview)
        Language.BE: "be-BY",
        Language.BE_BY: "be-BY",
        # Bengali (GA)
        Language.BN: "bn-BD",
        Language.BN_BD: "bn-BD",
        # Bulgarian (Preview)
        Language.BG: "bg-BG",
        Language.BG_BG: "bg-BG",
        # Burmese (Preview)
        Language.MY: "my-MM",
        Language.MY_MM: "my-MM",
        # Catalan (Preview)
        Language.CA: "ca-ES",
        Language.CA_ES: "ca-ES",
        # Cebuano (Preview)
        Language.CEB: "ceb-PH",
        Language.CEB_PH: "ceb-PH",
        # Chinese (Mandarin)
        Language.ZH: "cmn-CN",  # Preview
        Language.ZH_CN: "cmn-CN",
        Language.ZH_TW: "cmn-TW",  # Preview
        # Croatian (Preview)
        Language.HR: "hr-HR",
        Language.HR_HR: "hr-HR",
        # Czech (Preview)
        Language.CS: "cs-CZ",
        Language.CS_CZ: "cs-CZ",
        # Danish (Preview)
        Language.DA: "da-DK",
        Language.DA_DK: "da-DK",
        # Dutch (GA)
        Language.NL: "nl-NL",
        Language.NL_NL: "nl-NL",
        # English
        Language.EN: "en-US",  # GA
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",  # Preview
        Language.EN_GB: "en-GB",  # Preview
        Language.EN_IN: "en-IN",  # GA
        # Estonian (Preview)
        Language.ET: "et-EE",
        Language.ET_EE: "et-EE",
        # Filipino (Preview)
        Language.FIL: "fil-PH",
        Language.FIL_PH: "fil-PH",
        # Finnish (Preview)
        Language.FI: "fi-FI",
        Language.FI_FI: "fi-FI",
        # French
        Language.FR: "fr-FR",  # GA
        Language.FR_FR: "fr-FR",
        Language.FR_CA: "fr-CA",  # Preview
        # Galician (Preview)
        Language.GL: "gl-ES",
        Language.GL_ES: "gl-ES",
        # Georgian (Preview)
        Language.KA: "ka-GE",
        Language.KA_GE: "ka-GE",
        # German (GA)
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        # Greek (Preview)
        Language.EL: "el-GR",
        Language.EL_GR: "el-GR",
        # Gujarati (Preview)
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Haitian Creole (Preview)
        Language.HT: "ht-HT",
        Language.HT_HT: "ht-HT",
        # Hebrew (Preview)
        Language.HE: "he-IL",
        Language.HE_IL: "he-IL",
        # Hindi (GA)
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Hungarian (Preview)
        Language.HU: "hu-HU",
        Language.HU_HU: "hu-HU",
        # Icelandic (Preview)
        Language.IS: "is-IS",
        Language.IS_IS: "is-IS",
        # Indonesian (GA)
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        # Italian (GA)
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        # Japanese (GA)
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Javanese (Preview)
        Language.JV: "jv-JV",
        Language.JV_JV: "jv-JV",
        # Kannada (Preview)
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Konkani (Preview)
        Language.KOK: "kok-IN",
        Language.KOK_IN: "kok-IN",
        # Korean (GA)
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Lao (Preview)
        Language.LO: "lo-LA",
        Language.LO_LA: "lo-LA",
        # Latin (Preview)
        Language.LA: "la-VA",
        Language.LA_VA: "la-VA",
        # Latvian (Preview)
        Language.LV: "lv-LV",
        Language.LV_LV: "lv-LV",
        # Lithuanian (Preview)
        Language.LT: "lt-LT",
        Language.LT_LT: "lt-LT",
        # Luxembourgish (Preview)
        Language.LB: "lb-LU",
        Language.LB_LU: "lb-LU",
        # Macedonian (Preview)
        Language.MK: "mk-MK",
        Language.MK_MK: "mk-MK",
        # Maithili (Preview)
        Language.MAI: "mai-IN",
        Language.MAI_IN: "mai-IN",
        # Malagasy (Preview)
        Language.MG: "mg-MG",
        Language.MG_MG: "mg-MG",
        # Malay (Preview)
        Language.MS: "ms-MY",
        Language.MS_MY: "ms-MY",
        # Malayalam (Preview)
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Marathi (GA)
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Mongolian (Preview)
        Language.MN: "mn-MN",
        Language.MN_MN: "mn-MN",
        # Nepali (Preview)
        Language.NE: "ne-NP",
        Language.NE_NP: "ne-NP",
        # Norwegian
        Language.NO: "nb-NO",  # Preview: Bokmål
        Language.NB: "nb-NO",
        Language.NB_NO: "nb-NO",
        Language.NN: "nn-NO",  # Preview: Nynorsk
        Language.NN_NO: "nn-NO",
        # Odia (Preview)
        Language.OR: "or-IN",
        Language.OR_IN: "or-IN",
        # Pashto (Preview)
        Language.PS: "ps-AF",
        Language.PS_AF: "ps-AF",
        # Persian (Preview)
        Language.FA: "fa-IR",
        Language.FA_IR: "fa-IR",
        # Polish (GA)
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese
        Language.PT: "pt-BR",  # GA: Brazil
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",  # Preview: Portugal
        # Punjabi (Preview)
        Language.PA: "pa-IN",
        Language.PA_IN: "pa-IN",
        # Romanian (GA)
        Language.RO: "ro-RO",
        Language.RO_RO: "ro-RO",
        # Russian (GA)
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Serbian (Preview)
        Language.SR: "sr-RS",
        Language.SR_RS: "sr-RS",
        # Sindhi (Preview)
        Language.SD: "sd-IN",
        Language.SD_IN: "sd-IN",
        # Sinhala (Preview)
        Language.SI: "si-LK",
        Language.SI_LK: "si-LK",
        # Slovak (Preview)
        Language.SK: "sk-SK",
        Language.SK_SK: "sk-SK",
        # Slovenian (Preview)
        Language.SL: "sl-SI",
        Language.SL_SI: "sl-SI",
        # Spanish
        Language.ES: "es-ES",  # GA
        Language.ES_ES: "es-ES",
        Language.ES_419: "es-419",  # Preview: Latin America
        Language.ES_MX: "es-MX",  # Preview: Mexico
        # Swahili (Preview)
        Language.SW: "sw-KE",
        Language.SW_KE: "sw-KE",
        # Swedish (Preview)
        Language.SV: "sv-SE",
        Language.SV_SE: "sv-SE",
        # Tamil (GA)
        Language.TA: "ta-IN",
        Language.TA_IN: "ta-IN",
        # Telugu (GA)
        Language.TE: "te-IN",
        Language.TE_IN: "te-IN",
        # Thai (GA)
        Language.TH: "th-TH",
        Language.TH_TH: "th-TH",
        # Turkish (GA)
        Language.TR: "tr-TR",
        Language.TR_TR: "tr-TR",
        # Ukrainian (GA)
        Language.UK: "uk-UA",
        Language.UK_UA: "uk-UA",
        # Urdu (Preview)
        Language.UR: "ur-PK",
        Language.UR_PK: "ur-PK",
        # Vietnamese (GA)
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class GoogleHttpTTSService(TTSService):
    """Google Cloud Text-to-Speech HTTP service with SSML support.

    Provides text-to-speech synthesis using Google Cloud's HTTP API with
    comprehensive SSML support for voice customization, prosody control,
    and styling options. Ideal for applications requiring fine-grained
    control over speech output.

    Note:
        Requires Google Cloud credentials via service account JSON, credentials file,
        or default application credentials (GOOGLE_APPLICATION_CREDENTIALS).
        Chirp and Journey voices don't support SSML and will use plain text input.
    """

    class InputParams(BaseModel):
        """Input parameters for Google HTTP TTS voice customization.

        Parameters:
            pitch: Voice pitch adjustment (e.g., "+2st", "-50%").
            rate: Speaking rate adjustment (e.g., "slow", "fast", "125%"). Used for SSML prosody tags (non-Chirp voices).
            speaking_rate: Speaking rate for AudioConfig (Chirp/Journey voices). Range [0.25, 2.0].
            volume: Volume adjustment (e.g., "loud", "soft", "+6dB").
            emphasis: Emphasis level for the text.
            language: Language for synthesis. Defaults to English.
            gender: Voice gender preference.
            google_style: Google-specific voice style.
        """

        pitch: Optional[str] = None
        rate: Optional[str] = None
        speaking_rate: Optional[float] = None
        volume: Optional[str] = None
        emphasis: Optional[Literal["strong", "moderate", "reduced", "none"]] = None
        language: Optional[Language] = Language.EN
        gender: Optional[Literal["male", "female", "neutral"]] = None
        google_style: Optional[Literal["apologetic", "calm", "empathetic", "firm", "lively"]] = None

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        voice_id: str = "en-US-Chirp3-HD-Charon",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initializes the Google HTTP TTS service.

        Args:
            credentials: JSON string containing Google Cloud service account credentials.
            credentials_path: Path to Google Cloud service account JSON file.
            voice_id: Google TTS voice identifier (e.g., "en-US-Standard-A").
            sample_rate: Audio sample rate in Hz. If None, uses default.
            params: Voice customization parameters including pitch, rate, volume, etc.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or GoogleHttpTTSService.InputParams()

        self._settings = {
            "pitch": params.pitch,
            "rate": params.rate,
            "speaking_rate": params.speaking_rate,
            "volume": params.volume,
            "emphasis": params.emphasis,
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
            "gender": params.gender,
            "google_style": params.google_style,
        }
        self.set_voice(voice_id)
        self._client: texttospeech_v1.TextToSpeechAsyncClient = self._create_client(
            credentials, credentials_path
        )

    def _create_client(
        self, credentials: Optional[str], credentials_path: Optional[str]
    ) -> texttospeech_v1.TextToSpeechAsyncClient:
        """Create authenticated Google Text-to-Speech client.

        Args:
            credentials: JSON string with service account credentials.
            credentials_path: Path to service account JSON file.

        Returns:
            Authenticated TextToSpeechAsyncClient instance.

        Raises:
            ValueError: If no valid credentials are provided.
        """
        creds: Optional[service_account.Credentials] = None

        if credentials:
            # Use provided credentials JSON string
            json_account_info = json.loads(credentials)
            creds = service_account.Credentials.from_service_account_info(json_account_info)
        elif credentials_path:
            # Use service account JSON file if provided
            creds = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            try:
                creds, project_id = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except GoogleAuthError:
                pass

        if not creds:
            raise ValueError("No valid credentials provided.")

        return texttospeech_v1.TextToSpeechAsyncClient(credentials=creds)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Google HTTP TTS service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Google TTS language format.

        Args:
            language: The language to convert.

        Returns:
            The Google TTS-specific language code, or None if not supported.
        """
        return language_to_google_tts_language(language)

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Override to handle speaking_rate updates for Chirp/Journey voices.

        Args:
            settings: Dictionary of settings to update. Can include 'speaking_rate' (float)
        """
        if "speaking_rate" in settings:
            rate_value = float(settings["speaking_rate"])
            if 0.25 <= rate_value <= 2.0:
                self._settings["speaking_rate"] = rate_value
            else:
                logger.warning(
                    f"Invalid speaking_rate value: {rate_value}. Must be between 0.25 and 2.0"
                )
        await super()._update_settings(settings)

    def _construct_ssml(self, text: str) -> str:
        ssml = "<speak>"

        # Voice tag
        voice_attrs = [f"name='{self._voice_id}'"]

        language = self._settings["language"]
        voice_attrs.append(f"language='{language}'")

        if self._settings["gender"]:
            voice_attrs.append(f"gender='{self._settings['gender']}'")
        ssml += f"<voice {' '.join(voice_attrs)}>"

        # Prosody tag
        prosody_attrs = []
        if self._settings["pitch"]:
            prosody_attrs.append(f"pitch='{self._settings['pitch']}'")
        if self._settings["rate"]:
            prosody_attrs.append(f"rate='{self._settings['rate']}'")
        if self._settings["volume"]:
            prosody_attrs.append(f"volume='{self._settings['volume']}'")

        if prosody_attrs:
            ssml += f"<prosody {' '.join(prosody_attrs)}>"

        # Emphasis tag
        if self._settings["emphasis"]:
            ssml += f"<emphasis level='{self._settings['emphasis']}'>"

        # Google style tag
        if self._settings["google_style"]:
            ssml += f"<google:style name='{self._settings['google_style']}'>"

        ssml += text

        # Close tags
        if self._settings["google_style"]:
            ssml += "</google:style>"
        if self._settings["emphasis"]:
            ssml += "</emphasis>"
        if prosody_attrs:
            ssml += "</prosody>"
        ssml += "</voice></speak>"

        return ssml

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Google's HTTP TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            # Check if the voice is a Chirp voice (including Chirp 3) or Journey voice
            is_chirp_voice = "chirp" in self._voice_id.lower()
            is_journey_voice = "journey" in self._voice_id.lower()

            # Create synthesis input based on voice_id
            if is_chirp_voice or is_journey_voice:
                # Chirp and Journey voices don't support SSML, use plain text
                synthesis_input = texttospeech_v1.SynthesisInput(text=text)
            else:
                ssml = self._construct_ssml(text)
                synthesis_input = texttospeech_v1.SynthesisInput(ssml=ssml)

            voice = texttospeech_v1.VoiceSelectionParams(
                language_code=self._settings["language"], name=self._voice_id
            )
            # Build audio config with conditional speaking_rate
            audio_config_params = {
                "audio_encoding": texttospeech_v1.AudioEncoding.LINEAR16,
                "sample_rate_hertz": self.sample_rate,
            }

            # For Chirp and Journey voices, include speaking_rate in AudioConfig
            if (is_chirp_voice or is_journey_voice) and self._settings["speaking_rate"] is not None:
                audio_config_params["speaking_rate"] = self._settings["speaking_rate"]

            audio_config = texttospeech_v1.AudioConfig(**audio_config_params)

            request = texttospeech_v1.SynthesizeSpeechRequest(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            response = await self._client.synthesize_speech(request=request)

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Skip the first 44 bytes to remove the WAV header
            audio_content = response.audio_content[44:]

            CHUNK_SIZE = self.chunk_size

            for i in range(0, len(audio_content), CHUNK_SIZE):
                chunk = audio_content[i : i + CHUNK_SIZE]
                if not chunk:
                    break
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                yield frame

            yield TTSStoppedFrame()

        except Exception as e:
            error_message = f"TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)


class GoogleBaseTTSService(TTSService):
    """Base class for Google Cloud Text-to-Speech streaming services.

    Provides shared streaming synthesis logic for Google TTS services.
    This is an abstract base class. Use GoogleTTSService or GeminiTTSService instead.
    """

    def _create_client(
        self, credentials: Optional[str], credentials_path: Optional[str]
    ) -> texttospeech_v1.TextToSpeechAsyncClient:
        """Create authenticated Google Text-to-Speech client.

        Args:
            credentials: JSON string with service account credentials.
            credentials_path: Path to service account JSON file.

        Returns:
            Authenticated TextToSpeechAsyncClient instance.

        Raises:
            ValueError: If no valid credentials are provided.
        """
        creds: Optional[service_account.Credentials] = None

        if credentials:
            # Use provided credentials JSON string
            json_account_info = json.loads(credentials)
            creds = service_account.Credentials.from_service_account_info(json_account_info)
        elif credentials_path:
            # Use service account JSON file if provided
            creds = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            try:
                creds, project_id = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except GoogleAuthError:
                pass

        if not creds:
            raise ValueError("No valid credentials provided.")

        return texttospeech_v1.TextToSpeechAsyncClient(credentials=creds)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Google streaming TTS services support metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Google TTS language format.

        Args:
            language: The language to convert.

        Returns:
            The Google TTS-specific language code, or None if not supported.
        """
        return language_to_google_tts_language(language)

    async def _stream_tts(
        self,
        streaming_config: texttospeech_v1.StreamingSynthesizeConfig,
        text: str,
        prompt: Optional[str] = None,
    ) -> AsyncGenerator[Frame, None]:
        """Shared streaming synthesis logic.

        Args:
            streaming_config: The streaming configuration.
            text: The text to synthesize.
            prompt: Optional prompt for style instructions (Gemini only).

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        config_request = texttospeech_v1.StreamingSynthesizeRequest(
            streaming_config=streaming_config
        )

        async def request_generator():
            yield config_request
            synthesis_input_params = {"text": text}
            if prompt is not None:
                synthesis_input_params["prompt"] = prompt
            yield texttospeech_v1.StreamingSynthesizeRequest(
                input=texttospeech_v1.StreamingSynthesisInput(**synthesis_input_params)
            )

        streaming_responses = await self._client.streaming_synthesize(request_generator())
        await self.start_tts_usage_metrics(text)

        yield TTSStartedFrame()

        audio_buffer = b""
        first_chunk_for_ttfb = False

        CHUNK_SIZE = self.chunk_size

        async for response in streaming_responses:
            chunk = response.audio_content
            if not chunk:
                continue

            if not first_chunk_for_ttfb:
                await self.stop_ttfb_metrics()
                first_chunk_for_ttfb = True

            audio_buffer += chunk
            while len(audio_buffer) >= CHUNK_SIZE:
                piece = audio_buffer[:CHUNK_SIZE]
                audio_buffer = audio_buffer[CHUNK_SIZE:]
                yield TTSAudioRawFrame(piece, self.sample_rate, 1)

        if audio_buffer:
            yield TTSAudioRawFrame(audio_buffer, self.sample_rate, 1)

        yield TTSStoppedFrame()


class GoogleTTSService(GoogleBaseTTSService):
    """Google Cloud Text-to-Speech streaming service.

    Provides real-time text-to-speech synthesis using Google Cloud's streaming API
    for low-latency applications. Optimized for Chirp 3 HD and Journey voices
    with continuous audio streaming capabilities.

    Note:
        Requires Google Cloud credentials via service account JSON, file path, or
        default application credentials (GOOGLE_APPLICATION_CREDENTIALS env var).
        Only Chirp 3 HD and Journey voices are supported. Use GoogleHttpTTSService for other voices.

    Example::

        tts = GoogleTTSService(
            credentials_path="/path/to/service-account.json",
            voice_id="en-US-Chirp3-HD-Charon",
            params=GoogleTTSService.InputParams(
                language=Language.EN_US,
            )
        )
    """

    class InputParams(BaseModel):
        """Input parameters for Google streaming TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speaking_rate: The speaking rate, in the range [0.25, 2.0].
        """

        language: Optional[Language] = Language.EN
        speaking_rate: Optional[float] = None

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        voice_id: str = "en-US-Chirp3-HD-Charon",
        voice_cloning_key: Optional[str] = None,
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initializes the Google streaming TTS service.

        Args:
            credentials: JSON string containing Google Cloud service account credentials.
            credentials_path: Path to Google Cloud service account JSON file.
            voice_id: Google TTS voice identifier (e.g., "en-US-Chirp3-HD-Charon").
            voice_cloning_key: The voice cloning key for Chirp 3 custom voices.
            sample_rate: Audio sample rate in Hz. If None, uses default.
            params: Language configuration parameters.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or GoogleTTSService.InputParams()

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
            "speaking_rate": params.speaking_rate,
        }
        self.set_voice(voice_id)
        self._voice_cloning_key = voice_cloning_key
        self._client: texttospeech_v1.TextToSpeechAsyncClient = self._create_client(
            credentials, credentials_path
        )

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Override to handle speaking_rate updates for streaming API.

        Args:
            settings: Dictionary of settings to update. Can include 'speaking_rate' (float)
        """
        if "speaking_rate" in settings:
            rate_value = float(settings["speaking_rate"])
            if 0.25 <= rate_value <= 2.0:
                self._settings["speaking_rate"] = rate_value
            else:
                logger.warning(
                    f"Invalid speaking_rate value: {rate_value}. Must be between 0.25 and 2.0"
                )
        await super()._update_settings(settings)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate streaming speech from text using Google's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech as it's generated.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            # Build voice selection params
            if self._voice_cloning_key:
                voice_clone_params = texttospeech_v1.VoiceCloneParams(
                    voice_cloning_key=self._voice_cloning_key
                )
                voice = texttospeech_v1.VoiceSelectionParams(
                    language_code=self._settings["language"], voice_clone=voice_clone_params
                )
            else:
                voice = texttospeech_v1.VoiceSelectionParams(
                    language_code=self._settings["language"], name=self._voice_id
                )

            # Create streaming config
            streaming_config = texttospeech_v1.StreamingSynthesizeConfig(
                voice=voice,
                streaming_audio_config=texttospeech_v1.StreamingAudioConfig(
                    audio_encoding=texttospeech_v1.AudioEncoding.PCM,
                    sample_rate_hertz=self.sample_rate,
                    speaking_rate=self._settings["speaking_rate"],
                ),
            )

            # Use base class streaming logic
            async for frame in self._stream_tts(streaming_config, text):
                yield frame

        except Exception as e:
            await self.push_error(error_msg=f"TTS generation error: {str(e)}", exception=e)


class GeminiTTSService(GoogleBaseTTSService):
    """Gemini Text-to-Speech streaming service using Gemini TTS models.

    Provides real-time text-to-speech synthesis using Gemini's TTS-specific models
    (gemini-2.5-flash-tts and gemini-2.5-pro-tts) with support for natural
    voice control, prompts for style instructions, expressive markup tags,
    and multi-speaker conversations.

    Note:
        Requires Google Cloud credentials via service account JSON, credentials file,
        or default application credentials (GOOGLE_APPLICATION_CREDENTIALS).

        Uses the Google Cloud Text-to-Speech streaming API for low-latency synthesis.

    Example::

        tts = GeminiTTSService(
            credentials_path="/path/to/service-account.json",
            model="gemini-2.5-flash-tts",
            voice_id="Kore",
            params=GeminiTTSService.InputParams(
                language=Language.EN_US,
                prompt="Say this in a friendly and helpful tone"
            )
        )
    """

    GOOGLE_SAMPLE_RATE = 24000  # Google TTS always outputs at 24kHz

    # List of available Gemini TTS voices
    AVAILABLE_VOICES = [
        "Achernar",
        "Achird",
        "Algenib",
        "Algieba",
        "Alnilam",
        "Aoede",
        "Autonoe",
        "Callirhoe",
        "Charon",
        "Despina",
        "Enceladus",
        "Erinome",
        "Fenrir",
        "Gacrux",
        "Iapetus",
        "Kore",
        "Laomedeia",
        "Leda",
        "Orus",
        "Puck",
        "Pulcherrima",
        "Rasalgethi",
        "Sadachbia",
        "Sadaltager",
        "Schedar",
        "Sulafar",
        "Umbriel",
        "Vindemiatrix",
        "Zephyr",
        "Zubenelgenubi",
    ]

    class InputParams(BaseModel):
        """Input parameters for Gemini TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            prompt: Optional style instructions for how to synthesize the content.
            multi_speaker: Whether to enable multi-speaker support.
            speaker_configs: List of speaker configurations for multi-speaker mode.
        """

        language: Optional[Language] = Language.EN
        prompt: Optional[str] = None
        multi_speaker: bool = False
        speaker_configs: Optional[List[dict]] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-tts",
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        voice_id: str = "Kore",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initializes the Gemini TTS service.

        Args:
            api_key:

                .. deprecated:: 0.0.95
                    The `api_key` parameter is deprecated. Use `credentials` or
                    `credentials_path` instead for Google Cloud authentication.

            model: Gemini TTS model to use. Must be a TTS model like
                   "gemini-2.5-flash-tts" or "gemini-2.5-pro-tts".
            credentials: JSON string containing Google Cloud service account credentials.
            credentials_path: Path to Google Cloud service account JSON file.
            voice_id: Voice name from the available Gemini voices.
            sample_rate: Audio sample rate in Hz. If None, uses Google's default 24kHz.
            params: TTS configuration parameters.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        # Handle deprecated api_key parameter
        if api_key is not None:
            warnings.warn(
                "The 'api_key' parameter is deprecated and will be removed in a future version. "
                "Use 'credentials' or 'credentials_path' instead for Google Cloud authentication.",
                DeprecationWarning,
                stacklevel=2,
            )

        if sample_rate and sample_rate != self.GOOGLE_SAMPLE_RATE:
            logger.warning(
                f"Google TTS only supports {self.GOOGLE_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {sample_rate}Hz may cause issues."
            )
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or GeminiTTSService.InputParams()

        if voice_id not in self.AVAILABLE_VOICES:
            logger.warning(f"Voice '{voice_id}' not in known voices list. Using anyway.")

        self._model = model
        self._voice_id = voice_id
        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
            "prompt": params.prompt,
            "multi_speaker": params.multi_speaker,
            "speaker_configs": params.speaker_configs,
        }

        self._client: texttospeech_v1.TextToSpeechAsyncClient = self._create_client(
            credentials, credentials_path
        )

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Gemini TTS language format.

        Args:
            language: The language to convert.

        Returns:
            The Gemini TTS-specific language code, or None if not supported.
        """
        return language_to_gemini_tts_language(language)

    def set_voice(self, voice_id: str):
        """Set the voice for TTS generation.

        Args:
            voice_id: Name of the voice to use from AVAILABLE_VOICES.
        """
        if voice_id not in self.AVAILABLE_VOICES:
            logger.warning(f"Voice '{voice_id}' not in known voices list. Using anyway.")
        self._voice_id = voice_id

    async def start(self, frame: StartFrame):
        """Start the Gemini TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self.sample_rate != self.GOOGLE_SAMPLE_RATE:
            logger.warning(
                f"Google TTS requires {self.GOOGLE_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {self.sample_rate}Hz may cause issues."
            )

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Override to handle prompt updates.

        Args:
            settings: Dictionary of settings to update. Can include 'prompt' (str)
        """
        if "prompt" in settings:
            self._settings["prompt"] = settings["prompt"]
        await super()._update_settings(settings)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate streaming speech from text using Gemini TTS models.

        Args:
            text: The text to synthesize into speech. Can include markup tags
                  like [sigh], [laughing], [whispering] for expressive control.

        Yields:
            Frame: Audio frames containing the synthesized speech as it's generated.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            # Build voice selection params
            if self._settings["multi_speaker"] and self._settings["speaker_configs"]:
                # Multi-speaker mode
                speaker_voice_configs = []
                for speaker_config in self._settings["speaker_configs"]:
                    speaker_voice_configs.append(
                        texttospeech_v1.MultispeakerPrebuiltVoice(
                            speaker_alias=speaker_config["speaker_alias"],
                            speaker_id=speaker_config.get("speaker_id", self._voice_id),
                        )
                    )

                multi_speaker_voice_config = texttospeech_v1.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_voice_configs
                )

                voice = texttospeech_v1.VoiceSelectionParams(
                    language_code=self._settings["language"],
                    model_name=self._model,
                    multi_speaker_voice_config=multi_speaker_voice_config,
                )
            else:
                # Single speaker mode
                voice = texttospeech_v1.VoiceSelectionParams(
                    language_code=self._settings["language"],
                    name=self._voice_id,
                    model_name=self._model,
                )

            # Create streaming config
            streaming_config = texttospeech_v1.StreamingSynthesizeConfig(
                voice=voice,
                streaming_audio_config=texttospeech_v1.StreamingAudioConfig(
                    audio_encoding=texttospeech_v1.AudioEncoding.PCM,
                    sample_rate_hertz=self.sample_rate,
                ),
            )

            # Use base class streaming logic with prompt support
            async for frame in self._stream_tts(streaming_config, text, self._settings["prompt"]):
                yield frame

        except Exception as e:
            error_message = f"Gemini TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)
