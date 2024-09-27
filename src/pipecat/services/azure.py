#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import io

from typing import AsyncGenerator, Optional

from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    URLImageRawFrame,
)
from pipecat.services.ai_services import ImageGenService, STTService, TTSService
from pipecat.services.openai import BaseOpenAILLMService
from pipecat.utils.time import time_now_iso8601

from PIL import Image

from loguru import logger

# See .env.example for Azure configuration needed
try:
    from azure.cognitiveservices.speech import (
        CancellationReason,
        ResultReason,
        SpeechConfig,
        SpeechRecognizer,
        SpeechSynthesizer,
    )
    from azure.cognitiveservices.speech.audio import (
        AudioStreamFormat,
        PushAudioInputStream,
    )
    from azure.cognitiveservices.speech.dialog import AudioConfig
    from openai import AsyncAzureOpenAI
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Azure, you need to `pip install pipecat-ai[azure]`. Also, set `AZURE_SPEECH_API_KEY` and `AZURE_SPEECH_REGION` environment variables."
    )
    raise Exception(f"Missing module: {e}")


class AzureLLMService(BaseOpenAILLMService):
    def __init__(
        self, *, api_key: str, endpoint: str, model: str, api_version: str = "2023-12-01-preview"
    ):
        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version
        super().__init__(api_key=api_key, model=model)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )


class AzureTTSService(TTSService):
    class InputParams(BaseModel):
        emphasis: Optional[str] = None
        language: Optional[str] = "en-US"
        pitch: Optional[str] = None
        rate: Optional[str] = "1.05"
        role: Optional[str] = None
        style: Optional[str] = None
        style_degree: Optional[str] = None
        volume: Optional[str] = None

    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        voice="en-US-SaraNeural",
        sample_rate: int = 16000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        speech_config = SpeechConfig(subscription=api_key, region=region)
        self._speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        self._voice = voice
        self._sample_rate = sample_rate
        self._params = params

    def can_generate_metrics(self) -> bool:
        return True

    def _construct_ssml(self, text: str) -> str:
        ssml = (
            f"<speak version='1.0' xml:lang='{self._params.language}' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{self._voice}'>"
            "<mstts:silence type='Sentenceboundary' value='20ms' />"
        )

        if self._params.style:
            ssml += f"<mstts:express-as style='{self._params.style}'"
            if self._params.style_degree:
                ssml += f" styledegree='{self._params.style_degree}'"
            if self._params.role:
                ssml += f" role='{self._params.role}'"
            ssml += ">"

        prosody_attrs = []
        if self._params.rate:
            prosody_attrs.append(f"rate='{self._params.rate}'")
        if self._params.pitch:
            prosody_attrs.append(f"pitch='{self._params.pitch}'")
        if self._params.volume:
            prosody_attrs.append(f"volume='{self._params.volume}'")

        ssml += f"<prosody {' '.join(prosody_attrs)}>"

        if self._params.emphasis:
            ssml += f"<emphasis level='{self._params.emphasis}'>"

        ssml += text

        if self._params.emphasis:
            ssml += "</emphasis>"

        ssml += "</prosody>"

        if self._params.style:
            ssml += "</mstts:express-as>"

        ssml += "</voice></speak>"

        return ssml

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice = voice

    async def set_emphasis(self, emphasis: str):
        logger.debug(f"Setting TTS emphasis to: [{emphasis}]")
        self._params.emphasis = emphasis

    async def set_language(self, language: str):
        logger.debug(f"Setting TTS language code to: [{language}]")
        self._params.language = language

    async def set_pitch(self, pitch: str):
        logger.debug(f"Setting TTS pitch to: [{pitch}]")
        self._params.pitch = pitch

    async def set_rate(self, rate: str):
        logger.debug(f"Setting TTS rate to: [{rate}]")
        self._params.rate = rate

    async def set_role(self, role: str):
        logger.debug(f"Setting TTS role to: [{role}]")
        self._params.role = role

    async def set_style(self, style: str):
        logger.debug(f"Setting TTS style to: [{style}]")
        self._params.style = style

    async def set_style_degree(self, style_degree: str):
        logger.debug(f"Setting TTS style degree to: [{style_degree}]")
        self._params.style_degree = style_degree

    async def set_volume(self, volume: str):
        logger.debug(f"Setting TTS volume to: [{volume}]")
        self._params.volume = volume

    async def set_params(self, **kwargs):
        valid_params = {
            "voice": self.set_voice,
            "emphasis": self.set_emphasis,
            "language_code": self.set_language,
            "pitch": self.set_pitch,
            "rate": self.set_rate,
            "role": self.set_role,
            "style": self.set_style,
            "style_degree": self.set_style_degree,
            "volume": self.set_volume,
        }

        for param, value in kwargs.items():
            if param in valid_params:
                await valid_params[param](value)
            else:
                logger.warning(f"Ignoring unknown parameter: {param}")

        logger.debug(f"Updated TTS parameters: {', '.join(kwargs.keys())}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()

        ssml = self._construct_ssml(text)

        result = await asyncio.to_thread(self._speech_synthesizer.speak_ssml, (ssml))

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            await self.start_tts_usage_metrics(text)
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSStartedFrame())
            # Azure always sends a 44-byte header. Strip it off.
            yield TTSAudioRawFrame(
                audio=result.audio_data[44:], sample_rate=self._sample_rate, num_channels=1
            )
            await self.push_frame(TTSStoppedFrame())
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.warning(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                logger.error(f"{self} error: {cancellation_details.error_details}")


class AzureSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        language="en-US",
        sample_rate=16000,
        channels=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        speech_config = SpeechConfig(subscription=api_key, region=region)
        speech_config.speech_recognition_language = language

        stream_format = AudioStreamFormat(samples_per_second=sample_rate, channels=channels)
        self._audio_stream = PushAudioInputStream(stream_format)

        audio_config = AudioConfig(stream=self._audio_stream)
        self._speech_recognizer = SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        self._speech_recognizer.recognized.connect(self._on_handle_recognized)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        self._audio_stream.write(audio)
        await self.stop_processing_metrics()
        yield None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._speech_recognizer.start_continuous_recognition_async()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._speech_recognizer.stop_continuous_recognition_async()
        self._audio_stream.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._speech_recognizer.stop_continuous_recognition_async()
        self._audio_stream.close()

    def _on_handle_recognized(self, event):
        if event.result.reason == ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            frame = TranscriptionFrame(event.result.text, "", time_now_iso8601())
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())


class AzureImageGenServiceREST(ImageGenService):
    def __init__(
        self,
        *,
        image_size: str,
        api_key: str,
        endpoint: str,
        model: str,
        aiohttp_session: aiohttp.ClientSession,
        api_version="2023-06-01-preview",
    ):
        super().__init__()

        self._api_key = api_key
        self._azure_endpoint = endpoint
        self._api_version = api_version
        self.set_model_name(model)
        self._image_size = image_size
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        url = f"{self._azure_endpoint}openai/images/generations:submit?api-version={self._api_version}"

        headers = {"api-key": self._api_key, "Content-Type": "application/json"}

        body = {
            # Enter your prompt text here
            "prompt": prompt,
            "size": self._image_size,
            "n": 1,
        }

        async with self._aiohttp_session.post(url, headers=headers, json=body) as submission:
            # We never get past this line, because this header isn't
            # defined on a 429 response, but something is eating our
            # exceptions!
            operation_location = submission.headers["operation-location"]
            status = ""
            attempts_left = 120
            json_response = None
            while status != "succeeded":
                attempts_left -= 1
                if attempts_left == 0:
                    logger.error(f"{self} error: image generation timed out")
                    yield ErrorFrame("Image generation timed out")
                    return

                await asyncio.sleep(1)

                response = await self._aiohttp_session.get(operation_location, headers=headers)

                json_response = await response.json()
                status = json_response["status"]

            image_url = json_response["result"]["data"][0]["url"] if json_response else None
            if not image_url:
                logger.error(f"{self} error: image generation failed")
                yield ErrorFrame("Image generation failed")
                return

            # Load the image from the url
            async with self._aiohttp_session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                frame = URLImageRawFrame(
                    url=image_url, image=image.tobytes(), size=image.size, format=image.format
                )
                yield frame
