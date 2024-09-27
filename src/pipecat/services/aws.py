#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService

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
        language: Optional[str] = None
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
        sample_rate: int = 16000,
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
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._params = params

    def can_generate_metrics(self) -> bool:
        return True

    def _construct_ssml(self, text: str) -> str:
        ssml = "<speak>"

        if self._params.language:
            ssml += f"<lang xml:lang='{self._params.language}'>"

        prosody_attrs = []
        # Prosody tags are only supported for standard and neural engines
        if self._params.engine != "generative":
            if self._params.rate:
                prosody_attrs.append(f"rate='{self._params.rate}'")
            if self._params.pitch:
                prosody_attrs.append(f"pitch='{self._params.pitch}'")
            if self._params.volume:
                prosody_attrs.append(f"volume='{self._params.volume}'")

            if prosody_attrs:
                ssml += f"<prosody {' '.join(prosody_attrs)}>"
        else:
            logger.warning("Prosody tags are not supported for generative engine. Ignoring.")

        ssml += text

        if prosody_attrs:
            ssml += "</prosody>"

        if self._params.language:
            ssml += "</lang>"

        ssml += "</speak>"

        return ssml

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def set_engine(self, engine: str):
        logger.debug(f"Switching TTS engine to: [{engine}]")
        self._params.engine = engine

    async def set_language(self, language: str):
        logger.debug(f"Switching TTS language to: [{language}]")
        self._params.language = language

    async def set_pitch(self, pitch: str):
        logger.debug(f"Switching TTS pitch to: [{pitch}]")
        self._params.pitch = pitch

    async def set_rate(self, rate: str):
        logger.debug(f"Switching TTS rate to: [{rate}]")
        self._params.rate = rate

    async def set_volume(self, volume: str):
        logger.debug(f"Switching TTS volume to: [{volume}]")
        self._params.volume = volume

    async def set_params(self, params: InputParams):
        logger.debug(f"Switching TTS params to: [{params}]")
        self._params = params

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
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
                "Engine": self._params.engine,
                "SampleRate": str(self._sample_rate),
            }

            # Filter out None values
            filtered_params = {k: v for k, v in params.items() if v is not None}

            response = self._polly_client.synthesize_speech(**filtered_params)

            await self.start_tts_usage_metrics(text)

            await self.push_frame(TTSStartedFrame())

            if "AudioStream" in response:
                with response["AudioStream"] as stream:
                    audio_data = stream.read()
                    chunk_size = 8192
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i : i + chunk_size]
                        if len(chunk) > 0:
                            await self.stop_ttfb_metrics()
                            frame = TTSAudioRawFrame(chunk, self._sample_rate, 1)
                            yield frame

            await self.push_frame(TTSStoppedFrame())

        except (BotoCoreError, ClientError) as error:
            logger.exception(f"{self} error generating TTS: {error}")
            error_message = f"AWS Polly TTS error: {str(error)}"
            yield ErrorFrame(error=error_message)

        finally:
            await self.push_frame(TTSStoppedFrame())
