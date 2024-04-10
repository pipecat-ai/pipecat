from typing import Literal
import aiohttp
from PIL import Image
import io

from dailyai.services.ai_services import ImageGenService
from dailyai.services.openai_api_llm_service import BaseOpenAILLMService


try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use OpenAI, you need to `pip install dailyai[openai]`. Also, set `OPENAI_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class OpenAILLMService(BaseOpenAILLMService):

    def __init__(self, model="gpt-4", * args, **kwargs):
        super().__init__(model, *args, **kwargs)


class OpenAIImageGenService(ImageGenService):

    def __init__(
        self,
        *,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        aiohttp_session: aiohttp.ClientSession,
        api_key,
        model="dall-e-3",
    ):
        super().__init__()
        self._model = model
        self._image_size = image_size
        self._client = AsyncOpenAI(api_key=api_key)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> tuple[str, bytes, tuple[int, int]]:
        self.logger.info("Generating OpenAI image", prompt)

        image = await self._client.images.generate(
            prompt=prompt,
            model=self._model,
            n=1,
            size=self._image_size
        )
        image_url = image.data[0].url
        if not image_url:
            raise Exception("No image provided in response", image)

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            return (image_url, image.tobytes(), image.size)
