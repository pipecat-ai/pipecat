#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import io

from PIL import Image
from typing import AsyncGenerator

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    SystemFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    URLImageRawFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AsyncAIService, TTSService, ImageGenService
from pipecat.services.openai import BaseOpenAILLMService
from pipecat.utils.time import time_now_iso8601

from loguru import logger

# See .env.example for Azure configuration needed
try:
    from openai import AsyncAzureOpenAI
    from azure.cognitiveservices.speech import (
        SpeechConfig,
        SpeechRecognizer,
        SpeechSynthesizer,
        ResultReason,
        CancellationReason,
    )
    from azure.cognitiveservices.speech.audio import AudioStreamFormat, PushAudioInputStream
    from azure.cognitiveservices.speech.dialog import AudioConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Azure, you need to `pip install pipecat-ai[azure]`. Also, set `AZURE_SPEECH_API_KEY` and `AZURE_SPEECH_REGION` environment variables.")
    raise Exception(f"Missing module: {e}")


class AzureLLMService(BaseOpenAILLMService):
    def __init__(
            self,
            *,
            api_key: str,
            endpoint: str,
            model: str,
            api_version: str = "2023-12-01-preview"):
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
    def __init__(self, *, api_key: str, region: str, voice="en-US-SaraNeural", **kwargs):
        super().__init__(**kwargs)

        speech_config = SpeechConfig(subscription=api_key, region=region)
        self._speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        self._voice = voice

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()

        ssml = (
            "<speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{self._voice}'>"
            "<mstts:silence type='Sentenceboundary' value='20ms' />"
            "<mstts:express-as style='lyrical' styledegree='2' role='SeniorFemale'>"
            "<prosody rate='1.05'>"
            f"{text}"
            "</prosody></mstts:express-as></voice></speak> ")

        result = await asyncio.to_thread(self._speech_synthesizer.speak_ssml, (ssml))

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            await self.start_tts_usage_metrics(text)
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSStartedFrame())
            # Azure always sends a 44-byte header. Strip it off.
            yield AudioRawFrame(audio=result.audio_data[44:], sample_rate=16000, num_channels=1)
            await self.push_frame(TTSStoppedFrame())
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.warning(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                logger.error(f"{self} error: {cancellation_details.error_details}")


class AzureSTTService(AsyncAIService):
    def __init__(
            self,
            *,
            api_key: str,
            region: str,
            language="en-US",
            sample_rate=16000,
            channels=1,
            **kwargs):
        super().__init__(**kwargs)

        speech_config = SpeechConfig(subscription=api_key, region=region)
        speech_config.speech_recognition_language = language

        stream_format = AudioStreamFormat(samples_per_second=sample_rate, channels=channels)
        self._audio_stream = PushAudioInputStream(stream_format)

        audio_config = AudioConfig(stream=self._audio_stream)
        self._speech_recognizer = SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config)
        self._speech_recognizer.recognized.connect(self._on_handle_recognized)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            self._audio_stream.write(frame.audio)
        else:
            await self._push_queue.put((frame, direction))

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
            asyncio.run_coroutine_threadsafe(self.queue_frame(frame), self.get_event_loop())


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
        self._model = model
        self._image_size = image_size
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        url = f"{self._azure_endpoint}openai/images/generations:submit?api-version={self._api_version}"

        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"}

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
                    url=image_url,
                    image=image.tobytes(),
                    size=image.size,
                    format=image.format)
                yield frame
