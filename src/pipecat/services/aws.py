#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepgram, you need to `pip install pipecat-ai[aws]`. Also, set `AWS_SECRET_ACCESS_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_REGION` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def language_to_aws_language(language: Language) -> Optional[str]:
    language_map = {
        # Arabic
        Language.AR: "arb",
        Language.AR_AE: "ar-AE",
        # Catalan
        Language.CA: "ca-ES",
        # Chinese
        Language.ZH: "cmn-CN",  # Mandarin
        Language.YUE: "yue-CN",  # Cantonese
        Language.YUE_CN: "yue-CN",
        # Czech
        Language.CS: "cs-CZ",
        # Danish
        Language.DA: "da-DK",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_BE: "nl-BE",
        # English
        Language.EN: "en-US",  # Default to US English
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        Language.EN_NZ: "en-NZ",
        Language.EN_US: "en-US",
        Language.EN_ZA: "en-ZA",
        # Finnish
        Language.FI: "fi-FI",
        # French
        Language.FR: "fr-FR",
        Language.FR_BE: "fr-BE",
        Language.FR_CA: "fr-CA",
        # German
        Language.DE: "de-DE",
        Language.DE_AT: "de-AT",
        Language.DE_CH: "de-CH",
        # Hindi
        Language.HI: "hi-IN",
        # Icelandic
        Language.IS: "is-IS",
        # Italian
        Language.IT: "it-IT",
        # Japanese
        Language.JA: "ja-JP",
        # Korean
        Language.KO: "ko-KR",
        # Norwegian
        Language.NO: "nb-NO",
        Language.NB: "nb-NO",
        Language.NB_NO: "nb-NO",
        # Polish
        Language.PL: "pl-PL",
        # Portuguese
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        # Romanian
        Language.RO: "ro-RO",
        # Russian
        Language.RU: "ru-RU",
        # Spanish
        Language.ES: "es-ES",
        Language.ES_MX: "es-MX",
        Language.ES_US: "es-US",
        # Swedish
        Language.SV: "sv-SE",
        # Turkish
        Language.TR: "tr-TR",
        # Welsh
        Language.CY: "cy-GB",
        Language.CY_GB: "cy-GB",
    }

    return language_map.get(language)


class PollyTTSService(TTSService):
    class InputParams(BaseModel):
        engine: Optional[str] = None
        language: Optional[Language] = Language.EN
        pitch: Optional[str] = None
        rate: Optional[str] = None
        volume: Optional[str] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: Optional[str] = None,
        voice_id: str = "Joanna",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._polly_client = boto3.client(
            "polly",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=api_key,
            aws_session_token=aws_session_token,
            region_name=region,
        )
        self._settings = {
            "engine": params.engine,
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
            "pitch": params.pitch,
            "rate": params.rate,
            "volume": params.volume,
        }

        self._resampler = create_default_resampler()

        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_aws_language(language)

    def _construct_ssml(self, text: str) -> str:
        ssml = "<speak>"

        language = self._settings["language"]
        ssml += f"<lang xml:lang='{language}'>"

        prosody_attrs = []
        # Prosody tags are only supported for standard and neural engines
        if self._settings["engine"] != "generative":
            if self._settings["rate"]:
                prosody_attrs.append(f"rate='{self._settings['rate']}'")
            if self._settings["pitch"]:
                prosody_attrs.append(f"pitch='{self._settings['pitch']}'")
            if self._settings["volume"]:
                prosody_attrs.append(f"volume='{self._settings['volume']}'")

            if prosody_attrs:
                ssml += f"<prosody {' '.join(prosody_attrs)}>"
        else:
            logger.warning("Prosody tags are not supported for generative engine. Ignoring.")

        ssml += text

        if prosody_attrs:
            ssml += "</prosody>"

        ssml += "</lang>"

        ssml += "</speak>"

        return ssml

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        def read_audio_data(**args):
            response = self._polly_client.synthesize_speech(**args)
            if "AudioStream" in response:
                audio_data = response["AudioStream"].read()
                return audio_data
            return None

        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            # Construct the parameters dictionary
            ssml = self._construct_ssml(text)

            params = {
                "Text": ssml,
                "TextType": "ssml",
                "OutputFormat": "pcm",
                "VoiceId": self._voice_id,
                "Engine": self._settings["engine"],
                # AWS only supports 8000 and 16000 for PCM. We select 16000.
                "SampleRate": "16000",
            }

            # Filter out None values
            filtered_params = {k: v for k, v in params.items() if v is not None}

            audio_data = await asyncio.to_thread(read_audio_data, **filtered_params)

            if not audio_data:
                logger.error(f"{self} No audio data returned")
                yield None
                return

            audio_data = await self._resampler.resample(audio_data, 16000, self.sample_rate)

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    yield frame

            yield TTSStoppedFrame()

        except (BotoCoreError, ClientError) as error:
            logger.exception(f"{self} error generating TTS: {error}")
            error_message = f"AWS Polly TTS error: {str(error)}"
            yield ErrorFrame(error=error_message)

        finally:
            yield TTSStoppedFrame()
