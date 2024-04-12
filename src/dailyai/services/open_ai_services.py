from typing import Literal
import aiohttp
from PIL import Image
import io
import base64

from openai import AsyncOpenAI, AsyncStream

from openai.types.chat import (
    ChatCompletionChunk,
)

from dailyai.pipeline.frames import VisionImageFrame, TextFrame
from dailyai.services.ai_services import ImageGenService, VisionService
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


class OpenAIVisionService(VisionService):
    def __init__(
        self,
        *,
        model="gpt-4-vision-preview",
        api_key,
        base_url=None,
    ):
        self._model = model
        if base_url:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        super().__init__()

    async def run_vision_async(self, frame):
        prompt = frame.text
        IMAGE_WIDTH = frame.size[0]
        IMAGE_HEIGHT = frame.size[1]
        new_image = Image.frombytes(
            'RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), frame.image)

        # Uncomment these lines to write the frame to a jpg in the same directory.
        # current_path = os.getcwd()
        # image_path = os.path.join(current_path, "image.jpg")
        # image.save(image_path, format="JPEG")

        jpeg_buffer = io.BytesIO()

        new_image.save(jpeg_buffer, format='JPEG')

        jpeg_bytes = jpeg_buffer.getvalue()
        base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        chunks: AsyncStream[ChatCompletionChunk] = (
            await self._client.chat.completions.create(
                model=self._model,
                stream=True,
                messages=messages,
            )
        )
        async for chunk in chunks:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.content:
                yield TextFrame(chunk.choices[0].delta.content)
