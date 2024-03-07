import aiohttp
from PIL import Image
import io
import time
from openai import AsyncOpenAI

import json
from collections.abc import AsyncGenerator

from dailyai.services.ai_services import LLMService, ImageGenService


class OpenAILLMService(LLMService):
    def __init__(self, *, api_key, model="gpt-4", tools=None, messages=None):
        super().__init__(tools=tools, messages=messages)
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def get_response(self, messages, stream):
        return await self._client.chat.completions.create(
            stream=stream,
            messages=messages,
            model=self._model,
            tools=self._tools
        )

    async def run_llm_async(self, messages, tool_choice=None) -> AsyncGenerator[str, None]:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via openai: {messages_for_log}")
        if self._tools:
            tools = self._tools
        else:
            tools = None
        start_time = time.time()
        chunks = await self._client.chat.completions.create(model=self._model, stream=True, messages=messages, tools=tools, tool_choice=tool_choice)
        self.logger.info(f"=== OpenAI LLM TTFB: {time.time() - start_time}")
        async for chunk in chunks:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.tool_calls:
                yield chunk.choices[0].delta.tool_calls[0]
            elif chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def run_llm(self, messages) -> str | None:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via openai: {messages_for_log}")

        response = await self._client.chat.completions.create(model=self._model, stream=False, messages=messages)
        if response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None


class OpenAIImageGenService(ImageGenService):

    def __init__(
        self,
        *,
        image_size: str,
        aiohttp_session: aiohttp.ClientSession,
        api_key,
        model="dall-e-3",
    ):
        super().__init__(image_size=image_size)
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, sentence) -> tuple[str, bytes]:
        self.logger.info("Generating OpenAI image", sentence)

        image = await self._client.images.generate(
            prompt=sentence,
            model=self._model,
            n=1,
            size=self.image_size
        )
        image_url = image.data[0].url
        if not image_url:
            raise Exception("No image provided in response", image)

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            return (image_url, image.tobytes())
