#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import resample_audio
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


def language_to_aws_language(language: Language) -> str | None:
    language_map = {
        Language.CA: "ca-ES",
        Language.ZH: "cmn-CN",
        Language.DA: "da-DK",
        Language.NL: "nl-NL",
        Language.NL_BE: "nl-BE",
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_NZ: "en-NZ",
        Language.EN_IN: "en-IN",
        Language.FI: "fi-FI",
        Language.FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        Language.DE: "de-DE",
        Language.HI: "hi-IN",
        Language.IT: "it-IT",
        Language.JA: "ja-JP",
        Language.KO: "ko-KR",
        Language.NO: "nb-NO",
        Language.PL: "pl-PL",
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.RO: "ro-RO",
        Language.RU: "ru-RU",
        Language.ES: "es-ES",
        Language.SV: "sv-SE",
        Language.TR: "tr-TR",
    }
    return language_map.get(language, "en-US")


class AWSTTSService(TTSService):
    class InputParams(BaseModel):
        engine: Optional[str] = None
        language: Optional[Language] = Language.EN
        pitch: Optional[str] = None
        rate: Optional[str] = None
        volume: Optional[str] = None

    def __init__(
        self,
        *,
        api_key: str,
        aws_access_key_id: str,
        region: str,
        voice_id: str = "Joanna",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._polly_client = boto3.client(
            "polly",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=api_key,
            region_name=region,
        )
        self._settings = {
            "sample_rate": sample_rate,
            "engine": params.engine,
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
            "pitch": params.pitch,
            "rate": params.rate,
            "volume": params.volume,
        }

        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
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
                resampled = resample_audio(audio_data, 16000, self._settings["sample_rate"])
                return resampled
            return None

        logger.debug(f"Generating TTS: [{text}]")

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

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = TTSAudioRawFrame(chunk, self._settings["sample_rate"], 1)
                    yield frame

            yield TTSStoppedFrame()

        except (BotoCoreError, ClientError) as error:
            logger.exception(f"{self} error generating TTS: {error}")
            error_message = f"AWS Polly TTS error: {str(error)}"
            yield ErrorFrame(error=error_message)

        finally:
            yield TTSStoppedFrame()
