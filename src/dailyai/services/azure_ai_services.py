import aiohttp
import asyncio
import io
import json
from openai import AzureOpenAI

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
        self.speech_synthesizer = SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)

    async def run_tts(self, sentence) -> AsyncGenerator[bytes, None, None]:
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
        api_version = api_version or "2023-12-01-preview"
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.model = model or os.getenv("AZURE_CHATGPT_DEPLOYMENT_ID")

    def get_response(self, messages, stream):
        return self.client.chat.completions.create(
            stream=stream,
            messages=messages,
            model=self.model,
        )

    async def run_llm_async(self, messages) -> AsyncGenerator[str, None, None]:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via azure: {messages_for_log}")

        response = self.get_response(messages, stream=True)

        for chunk in response:
            if len(chunk.choices) == 0:
                continue

            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def run_llm(self, messages) -> str | None:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via azure: {messages_for_log}")

        response = self.get_response(messages, stream=False)
        if response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None

class AzureImageGenServiceREST(ImageGenService):

    def __init__(self, api_key=None, azure_endpoint=None, api_version=None, model=None):
        super().__init__()
        self.api_key = api_key or os.getenv("AZURE_DALLE_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_DALLE_ENDPOINT")
        self.api_version = api_version or "2023-06-01-preview"
        self.model = model or os.getenv("AZURE_DALLE_DEPLOYMENT_ID")

    async def run_image_gen(self, sentence, size) -> tuple[str, Image.Image]:
        # TODO hoist the session to app-level
        async with aiohttp.ClientSession() as session:
            url = f"{self.azure_endpoint}openai/images/generations:submit?api-version={self.api_version}"
            headers= { "api-key": self.api_key, "Content-Type": "application/json" }
            body = {
                # Enter your prompt text here
                "prompt": sentence,
                "size": size,
                "n": 1,
            }
            async with session.post(url, headers=headers, json=body) as submission:
                operation_location = submission.headers['operation-location']

                status = ""
                attempts_left = 120
                while status != "succeeded":
                    attempts_left -= 1
                    if attempts_left == 0:
                        raise Exception("Image generation timed out")

                    await asyncio.sleep(1)
                    response = await session.get(operation_location, headers=headers)
                    json_response = await response.json()
                    status = json_response["status"]

                image_url = json_response["result"]["data"][0]["url"]

                # Load the image from the url
                async with session.get(image_url) as response:
                    image_stream = io.BytesIO(await response.content.read())
                    image = Image.open(image_stream)
                    return (image_url, image.tobytes())


class AzureImageGenService(ImageGenService):

    def __init__(self, api_key=None, azure_endpoint=None, api_version=None, model=None):
        super().__init__()

        api_key = api_key or os.getenv("AZURE_DALLE_KEY")
        azure_endpoint = azure_endpoint or os.getenv("AZURE_DALLE_ENDPOINT")
        api_version = api_version or "2023-06-01-preview"
        self.model = model or os.getenv("AZURE_DALLE_DEPLOYMENT_ID")

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    async def run_image_gen(self, sentence) -> tuple[str, Image.Image]:
        self.logger.info("Generating azure image", sentence)

        image = self.client.images.generate(
            model=self.model,
            prompt=sentence,
            n=1,
            size=f"1024x1024",
        )

        url = image["data"][0]["url"]
        response = requests.get(url)

        dalle_stream = io.BytesIO(response.content)
        dalle_im = Image.open(dalle_stream)

        return (url, dalle_im)
