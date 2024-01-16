import fal
import aiohttp
import asyncio
import io
import json
from PIL import Image


from dailyai.services.ai_services import LLMService, TTSService, ImageGenService
# Fal expects FAL_KEY_ID and FAL_KEY_SECRET to be set in the env
class FalImageGenService(ImageGenService):
    def __init__(self):
        super().__init__()



    async def run_image_gen(self, sentence, size) -> tuple[str, bytes]:
        def get_image_url(sentence, size):
            handler = fal.apps.submit(
                "110602490-fast-sdxl",
                arguments={
                "prompt": sentence
                },
                )
            for event in handler.iter_events():
                if isinstance(event, fal.apps.InProgress):
                    print('Request in progress')
                    print(event.logs)

            result = handler.get()

            image_url = result["images"][0]["url"] if result else None
            if not image_url:
                raise Exception("Image generation failed")

            return image_url
        
        image_url = await asyncio.to_thread(get_image_url, sentence, size)
        # Load the image from the url
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                return (image_url, image.tobytes())

        # return (image_url, dalle_im.tobytes())