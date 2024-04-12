import aiohttp
import os

from typing import Literal

from dailyai.services.ai_services import ImageGenService, VisionService
from dailyai.services.openai_api_llm_service import BaseOpenAILLMService
from dailyai.services.open_ai_services import OpenAIVisionService

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use Fireworks, you need to `pip install dailyai[fireworks]`. Also, set the `FIREWORKS_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class FireworksLLMService(BaseOpenAILLMService):
    def __init__(self, model="accounts/fireworks/models/firefunction-v1", *args, **kwargs):
        kwargs["base_url"] = "https://api.fireworks.ai/inference/v1"
        super().__init__(model, *args, **kwargs)


class FireworksImageGenService(ImageGenService):

    def __init__(
        self,
        *,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        aiohttp_session: aiohttp.ClientSession,
        api_key,
        model="accounts/fireworks/models/stable-diffusion-xl-1024-v1-0",
    ):
        super().__init__()
        self._model = model
        self._image_size = image_size
        self._client = AsyncOpenAI(api_key=api_key,
                                   base_url="https://api.fireworks.ai/inference/v1")
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> tuple[str, bytes, tuple[int, int]]:
        self.logger.info(f"Generating Fireworks image: {prompt}")

        image = await self._client.images.generate(
            prompt=prompt,
            model=self._model,
            n=1,
            size=self._image_size
        )
        print(f"!!! image is {image}")
        image_url = image.data[0].url
        if not image_url:
            raise Exception("No image provided in response", image)

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            return (image_url, image.tobytes(), image.size)


class FireworksVisionService(OpenAIVisionService):
    def __init__(self, *, api_key, model="accounts/fireworks/models/firellava-13b"):
        super().__init__(model=model, api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
