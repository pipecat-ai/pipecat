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
