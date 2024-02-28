import fal
import aiohttp
import asyncio
import io
import os
from PIL import Image

from dailyai.services.ai_services import ImageGenService


from dailyai.services.ai_services import ImageGenService
# Fal expects FAL_KEY_ID and FAL_KEY_SECRET to be set in the env


class FalImageGenService(ImageGenService):
    def __init__(
            self,
            *,
            image_size,
            aiohttp_session: aiohttp.ClientSession,
            key_id=None,
            key_secret=None):
        super().__init__(image_size)
        self._aiohttp_session = aiohttp_session
        if key_id:
            os.environ["FAL_KEY_ID"] = key_id
        if key_secret:
            os.environ["FAL_KEY_SECRET"] = key_secret

    async def run_image_gen(self, sentence) -> tuple[str, bytes]:
        def get_image_url(sentence, size):
            handler = fal.apps.submit(
                "110602490-fast-sdxl",
                arguments={
                    "prompt": sentence
                },
            )
            for event in handler.iter_events():
                if isinstance(event, fal.apps.InProgress):
                    pass

            result = handler.get()

            image_url = result["images"][0]["url"] if result else None
            if not image_url:
                raise Exception("Image generation failed")

            return image_url
        image_url = await asyncio.to_thread(get_image_url, sentence, self.image_size)
        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            return (image_url, image.tobytes())
