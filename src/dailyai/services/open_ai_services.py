import aiohttp
from PIL import Image
import io
from openai import AsyncOpenAI

import os
import json
from collections.abc import AsyncGenerator

from dailyai.services.ai_services import LLMService, ImageGenService


class OpenAILLMService(LLMService):
    def __init__(self, api_key=None, model=None):
        super().__init__()
        api_key = api_key or os.getenv("OPEN_AI_KEY")
        self.model = model or os.getenv("OPEN_AI_LLM_MODEL") or "gpt-4"
        self.client = AsyncOpenAI(api_key=api_key)

    async def get_response(self, messages, stream):
        return await self.client.chat.completions.create(
            stream=stream,
            messages=messages,
            model=self.model
        )

    async def run_llm_async(self, messages) -> AsyncGenerator[str, None]:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via openai: {messages_for_log}")

        response = await self.get_response(messages, stream=True)

        for chunk in response:
            if len(chunk.choices) == 0:
                continue

            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def run_llm(self, messages) -> str | None:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via openai: {messages_for_log}")

        response = await self.get_response(messages, stream=False)
        if response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None

class OpenAIImageGenService(ImageGenService):

    def __init__(
        self,
        image_size: str,
        api_key=None,
        model=None,
        aiohttp_session: aiohttp.ClientSession | None = None,
    ):
        super().__init__(image_size=image_size)
        api_key = api_key or os.getenv("OPEN_AI_KEY")
        self.model = model or os.getenv("OPEN_AI_IMAGE_MODEL") or "dall-e-3"
        self.client = AsyncOpenAI(api_key=api_key)
        self.aiohttp_session=aiohttp_session or aiohttp.ClientSession()

    async def run_image_gen(self, sentence) -> tuple[str, bytes]:
        self.logger.info("Generating OpenAI image", sentence)

        image = await self.client.images.generate(
            prompt=sentence,
            model=self.model,
            n=1,
            size=self.image_size
        )
        image_url = image.data[0].url
        if not image_url:
            raise Exception("No image provided in response", image)

        # Load the image from the url
        async with self.aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            return (image_url, image.tobytes())
