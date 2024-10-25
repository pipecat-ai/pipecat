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
            else Language.EN,
            "pitch": params.pitch,
            "rate": params.rate,
            "volume": params.volume,
        }

        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        match language:
            case Language.CA:
                return "ca-ES"
            case Language.ZH:
                return "cmn-CN"
            case Language.DA:
                return "da-DK"
            case Language.NL:
                return "nl-NL"
            case Language.NL_BE:
                return "nl-BE"
            case Language.EN | Language.EN_US:
                return "en-US"
            case Language.EN_AU:
                return "en-AU"
            case Language.EN_GB:
                return "en-GB"
            case Language.EN_NZ:
                return "en-NZ"
            case Language.EN_IN:
                return "en-IN"
            case Language.FI:
                return "fi-FI"
            case Language.FR:
                return "fr-FR"
            case Language.FR_CA:
                return "fr-CA"
            case Language.DE:
                return "de-DE"
            case Language.HI:
                return "hi-IN"
            case Language.IT:
                return "it-IT"
            case Language.JA:
                return "ja-JP"
            case Language.KO:
                return "ko-KR"
            case Language.NO:
                return "nb-NO"
            case Language.PL:
                return "pl-PL"
            case Language.PT:
                return "pt-PT"
            case Language.PT_BR:
                return "pt-BR"
            case Language.RO:
                return "ro-RO"
            case Language.RU:
                return "ru-RU"
            case Language.ES:
                return "es-ES"
            case Language.SV:
                return "sv-SE"
            case Language.TR:
                return "tr-TR"
        return None

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
