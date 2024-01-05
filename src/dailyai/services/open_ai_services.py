import requests
import aiohttp
import asyncio
from PIL import Image
import io
from openai import AsyncOpenAI

import os
import json
from collections.abc import AsyncGenerator

from dailyai.services.ai_services import AIService, TTSService, LLMService, ImageGenService


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

    async def run_llm_async(self, messages) -> AsyncGenerator[str, None, None]:
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
    def __init__(self, api_key=None, model=None):
        super().__init__()
        api_key = api_key or os.getenv("OPEN_AI_KEY")
        self.model = model or os.getenv("OPEN_AI_IMAGE_MODEL") or "dall-e-3"
        self.client = AsyncOpenAI(api_key=api_key)

    async def run_image_gen(self, sentence, size) -> tuple[str, Image.Image]:
        self.logger.info("Generating OpenAI image", sentence)

        image = await self.client.images.generate(
            prompt=sentence,
            model=self.model,
            n=1,
            size=size
        )
        image_url = image.data[0].url
        response = requests.get(image_url)

        dalle_stream = io.BytesIO(response.content)
        dalle_im = Image.open(dalle_stream)

        return (image_url, dalle_im.tobytes())
