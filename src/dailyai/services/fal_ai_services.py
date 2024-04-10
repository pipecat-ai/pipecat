import aiohttp
import asyncio
import io
import os
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Union, Dict

from dailyai.services.ai_services import ImageGenService

try:
    import fal
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use Fal, you need to `pip install dailyai[fal]`. Also, set `FAL_KEY_ID` and `FAL_KEY_SECRET` environment variables.")
    raise Exception(f"Missing module: {e}")


class FalImageGenService(ImageGenService):
    class InputParams(BaseModel):
        seed: Optional[int] = None
        num_inference_steps: int = 4
        num_images: int = 1
        image_size: Union[str, Dict[str, int]] = "square_hd"
        expand_prompt: bool = False
        enable_safety_checker: bool = True
        format: str = "png"

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        params: InputParams,
        model="fal-ai/fast-sdxl",
        key_id=None,
        key_secret=None
    ):
        super().__init__()
        self._model = model
        self._params = params
        self._aiohttp_session = aiohttp_session
        if key_id:
            os.environ["FAL_KEY_ID"] = key_id
        if key_secret:
            os.environ["FAL_KEY_SECRET"] = key_secret

    async def run_image_gen(self, prompt: str) -> tuple[str, bytes, tuple[int, int]]:
        def get_image_url(prompt):
            handler = fal.apps.submit(  # type: ignore
                self._model,
                arguments={
                    "prompt": prompt,
                    **self._params.dict(),
                },
            )
            for event in handler.iter_events():
                if isinstance(event, fal.apps.InProgress):  # type: ignore
                    pass

            result = handler.get()

            image_url = result["images"][0]["url"] if result else None
            if not image_url:
                raise Exception("Image generation failed")

            return image_url

        image_url = await asyncio.to_thread(get_image_url, prompt)

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            return (image_url, image.tobytes(), image.size)
