import aiohttp
import asyncio
import io
import json
from openai import AsyncAzureOpenAI

import os
import requests

from collections.abc import AsyncGenerator

from dailyai.services.ai_services import LLMService, TTSService, ImageGenService
from PIL import Image

# See .env.example for Azure configuration needed
from azure.cognitiveservices.speech import SpeechSynthesizer, SpeechConfig, ResultReason, CancellationReason


class AzureTTSService(TTSService):
    def __init__(self, speech_key=None, speech_region=None):
        super().__init__()

        speech_key = speech_key or os.getenv("AZURE_SPEECH_SERVICE_KEY")
        speech_region = speech_region or os.getenv("AZURE_SPEECH_SERVICE_REGION")

        self.speech_config = SpeechConfig(subscription=speech_key, region=speech_region)
        self.speech_synthesizer = SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None)

    async def run_tts(self, sentence) -> AsyncGenerator[bytes, None]:
        self.logger.info("Running azure tts")
        ssml = "<speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis' " \
            "xmlns:mstts='http://www.w3.org/2001/mstts'>" \
            "<voice name='en-US-SaraNeural'>" \
            "<mstts:silence type='Sentenceboundary' value='20ms' />" \
            "<mstts:express-as style='lyrical' styledegree='2' role='SeniorFemale'>" \
            "<prosody rate='1.05'>" \
            f"{sentence}" \
            "</prosody></mstts:express-as></voice></speak> "
        result = await asyncio.to_thread(self.speech_synthesizer.speak_ssml, (ssml))
        self.logger.info("Got azure tts result")
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            self.logger.info("Returning result")
            # azure always sends a 44-byte header. Strip it off.
            yield result.audio_data[44:]
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            self.logger.info("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == CancellationReason.Error:
                self.logger.info("Error details: {}".format(cancellation_details.error_details))


class AzureLLMService(LLMService):
    def __init__(self, api_key=None, azure_endpoint=None, api_version=None, model=None):
        super().__init__()
        api_key = api_key or os.getenv("AZURE_CHATGPT_KEY")

        azure_endpoint = azure_endpoint or os.getenv("AZURE_CHATGPT_ENDPOINT")
        if not azure_endpoint:
            raise Exception(
                "No azure endpoint specified for Azure LLM, please set AZURE_CHATGPT_ENDPOINT in the environment or pass it to the AzureLLMService constructor")

        model: str | None = model or os.getenv("AZURE_CHATGPT_DEPLOYMENT_ID")
        if not model:
            raise Exception(
                "No model specified for Azure LLM, please set AZURE_CHATGPT_DEPLOYMENT_ID in the environment or pass it to the AzureLLMService constructor")
        self.model: str = model

        api_version = api_version or "2023-12-01-preview"
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    async def run_llm_async(self, messages) -> AsyncGenerator[str, None]:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via azure: {messages_for_log}")

        chunks = await self.client.chat.completions.create(model=self.model, stream=True, messages=messages)
        async for chunk in chunks:
            if len(chunk.choices) == 0:
                continue

            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def run_llm(self, messages) -> str | None:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via azure: {messages_for_log}")

        response = await self.client.chat.completions.create(model=self.model, stream=False, messages=messages)
        if response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None


class AzureImageGenServiceREST(ImageGenService):

    def __init__(
            self,
            image_size: str,
            aiohttp_session: aiohttp.ClientSession,
            api_key=None,
            azure_endpoint=None,
            api_version=None,
            model=None):
        super().__init__(image_size=image_size)
        self._api_key = api_key or os.getenv("AZURE_DALLE_KEY")
        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_DALLE_ENDPOINT")
        self._api_version = api_version or "2023-06-01-preview"
        self._model = model or os.getenv("AZURE_DALLE_DEPLOYMENT_ID")
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, sentence) -> tuple[str, bytes]:
        url = f"{self._azure_endpoint}openai/images/generations:submit?api-version={self._api_version}"
        headers = {"api-key": self._api_key, "Content-Type": "application/json"}
        body = {
            # Enter your prompt text here
            "prompt": sentence,
            "size": self.image_size,
            "n": 1,
        }
        async with self._aiohttp_session.post(
            url, headers=headers, json=body
        ) as submission:
            operation_location = submission.headers['operation-location']

            status = ""
            attempts_left = 120
            json_response = None
            while status != "succeeded":
                attempts_left -= 1
                if attempts_left == 0:
                    raise Exception("Image generation timed out")

                await asyncio.sleep(1)
                response = await self._aiohttp_session.get(
                    operation_location, headers=headers
                )
                json_response = await response.json()
                status = json_response["status"]

            image_url = json_response["result"]["data"][0]["url"] if json_response else None
            if not image_url:
                raise Exception("Image generation failed")

            # Load the image from the url
            async with self._aiohttp_session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                return (image_url, image.tobytes())
