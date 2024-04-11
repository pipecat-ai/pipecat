import aiohttp
import asyncio
import io
import os
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Union, Dict


from dailyai.services.ai_services import ImageGenService

try:
    import fal_client
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use Fal, you need to `pip install dailyai[fal]`. Also, set `FAL_KEY` environment variable.")
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
        key=None,
    ):
        super().__init__()
        self._model = model
        self._params = params
        self._aiohttp_session = aiohttp_session
        if key:
            os.environ["FAL_KEY"] = key

    async def run_image_gen(self, prompt: str) -> tuple[str, bytes, tuple[int, int]]:
        response = await fal_client.run_async(
            self._model,
            arguments={"prompt": prompt, **self._params.dict()}
        )

        image_url = response["images"][0]["url"] if response else None

        if not image_url:
            raise Exception("Image generation failed")

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            return (image_url, image.tobytes(), image.size)
