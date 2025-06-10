#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import os

from pipecat.utils.tracing.service_decorators import traced_tts

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from typing import AsyncGenerator, Literal, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

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
    language_map = {
        # Afrikaans
        Language.AF: "af-ZA",
        Language.AF_ZA: "af-ZA",
        # Arabic
        Language.AR: "ar-XA",
        # Bengali
        Language.BN: "bn-IN",
        Language.BN_IN: "bn-IN",
        # Bulgarian
        Language.BG: "bg-BG",
        Language.BG_BG: "bg-BG",
        # Catalan
        Language.CA: "ca-ES",
        Language.CA_ES: "ca-ES",
        # Chinese (Mandarin and Cantonese)
        Language.ZH: "cmn-CN",
        Language.ZH_CN: "cmn-CN",
        Language.ZH_TW: "cmn-TW",
        Language.ZH_HK: "yue-HK",
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
        # Filipino
        Language.FIL: "fil-PH",
        Language.FIL_PH: "fil-PH",
        # Finnish
        Language.FI: "fi-FI",
        Language.FI_FI: "fi-FI",
        # French
        Language.FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        Language.FR_FR: "fr-FR",
        # Galician
        Language.GL: "gl-ES",
        Language.GL_ES: "gl-ES",
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
        # Icelandic
        Language.IS: "is-IS",
        Language.IS_IS: "is-IS",
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
        # Malay
        Language.MS: "ms-MY",
        Language.MS_MY: "ms-MY",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
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
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        # Punjabi
        Language.PA: "pa-IN",
        Language.PA_IN: "pa-IN",
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
        # Spanish
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
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
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
    }

    return language_map.get(language)


class GoogleHttpTTSService(TTSService):
    class InputParams(BaseModel):
        pitch: Optional[str] = None
        rate: Optional[str] = None
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
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or GoogleHttpTTSService.InputParams()

        self._settings = {
            "pitch": params.pitch,
            "rate": params.rate,
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
        creds: Optional[service_account.Credentials] = None

        # Create a Google Cloud service account for the Cloud Text-to-Speech API
        # Using either the provided credentials JSON string or the path to a service account JSON
        # file, create a Google Cloud service account and use it to authenticate with the API.
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
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_google_tts_language(language)

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
            audio_config = texttospeech_v1.AudioConfig(
                audio_encoding=texttospeech_v1.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
            )

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
            logger.exception(f"{self} error generating TTS: {e}")
            error_message = f"TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)


class GoogleTTSService(TTSService):
    """Text-to-Speech service using Google Cloud Text-to-Speech API.

    Converts text to speech using Google's TTS models with streaming synthesis
    for low latency. Supports multiple languages and voices.

    Args:
        credentials: JSON string containing Google Cloud service account credentials.
        credentials_path: Path to Google Cloud service account JSON file.
        voice_id: Google TTS voice identifier (e.g., "en-US-Chirp3-HD-Charon").
        sample_rate: Audio sample rate in Hz.
        params: Language only.

    Notes:
        Requires Google Cloud credentials via service account JSON, file path, or
        default application credentials (GOOGLE_APPLICATION_CREDENTIALS env var).
        Only Chirp 3 HD and Journey voices are supported. Use GoogleHttpTTSService for other voices.

    Example:
        ```python
        tts = GoogleTTSService(
            credentials_path="/path/to/service-account.json",
            voice_id="en-US-Chirp3-HD-Charon",
            params=GoogleTTSService.InputParams(
                language=Language.EN_US,
            )
        )
        ```
    """

    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        voice_id: str = "en-US-Chirp3-HD-Charon",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or GoogleTTSService.InputParams()

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
        }
        self.set_voice(voice_id)
        self._client: texttospeech_v1.TextToSpeechAsyncClient = self._create_client(
            credentials, credentials_path
        )

    def _create_client(
        self, credentials: Optional[str], credentials_path: Optional[str]
    ) -> texttospeech_v1.TextToSpeechAsyncClient:
        creds: Optional[service_account.Credentials] = None

        # Create a Google Cloud service account for the Cloud Text-to-Speech API
        # Using either the provided credentials JSON string or the path to a service account JSON
        # file, create a Google Cloud service account and use it to authenticate with the API.
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
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_google_tts_language(language)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            voice = texttospeech_v1.VoiceSelectionParams(
                language_code=self._settings["language"], name=self._voice_id
            )

            streaming_config = texttospeech_v1.StreamingSynthesizeConfig(
                voice=voice,
                streaming_audio_config=texttospeech_v1.StreamingAudioConfig(
                    audio_encoding=texttospeech_v1.AudioEncoding.PCM,
                    sample_rate_hertz=self.sample_rate,
                ),
            )
            config_request = texttospeech_v1.StreamingSynthesizeRequest(
                streaming_config=streaming_config
            )

            async def request_generator():
                yield config_request
                yield texttospeech_v1.StreamingSynthesizeRequest(
                    input=texttospeech_v1.StreamingSynthesisInput(text=text)
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

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            error_message = f"TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)
