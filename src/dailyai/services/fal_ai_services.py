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
            print("starting fal submit...")
            handler = fal.apps.submit(
                "110602490-fast-sdxl",
                arguments={
                "prompt": sentence
                },
                )
            print("past fal handler init, about to wait for iter_events...")
            for event in handler.iter_events():
                if isinstance(event, fal.apps.InProgress):
                    print('Request in progress')
                    print(event.logs)

            result = handler.get()

            image_url = result["images"][0]["url"] if result else None
            if not image_url:
                raise Exception("Image generation failed")

            return image_url
        print(f"fetching image url...")
        image_url = await asyncio.to_thread(get_image_url, sentence, size)
        print(f"got image url, downloading image...")
        # Load the image from the url
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                print("got image response")
                image_stream = io.BytesIO(await response.content.read())
                print("read image stream")
                image = Image.open(image_stream)
                return (image_url, image.tobytes())

        # return (image_url, dalle_im.tobytes())