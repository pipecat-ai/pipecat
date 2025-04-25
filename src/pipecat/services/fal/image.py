#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import os
from typing import AsyncGenerator, Dict, Optional, Union

import aiohttp
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.image_service import ImageGenService

try:
    import fal_client
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Fal, you need to `pip install pipecat-ai[fal]`.")
    raise Exception(f"Missing module: {e}")


class FalImageGenService(ImageGenService):
    class InputParams(BaseModel):
        seed: Optional[int] = None
        num_inference_steps: int = 8
        num_images: int = 1
        image_size: Union[str, Dict[str, int]] = "square_hd"
        expand_prompt: bool = False
        enable_safety_checker: bool = True
        format: str = "png"

    def __init__(
        self,
        *,
        params: InputParams,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "fal-ai/fast-sdxl",
        key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._params = params
        self._aiohttp_session = aiohttp_session
        if key:
            os.environ["FAL_KEY"] = key

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        def load_image_bytes(encoded_image: bytes):
            buffer = io.BytesIO(encoded_image)
            image = Image.open(buffer)
            return (image.tobytes(), image.size, image.format)

        logger.debug(f"Generating image from prompt: {prompt}")

        response = await fal_client.run_async(
            self.model_name,
            arguments={"prompt": prompt, **self._params.model_dump(exclude_none=True)},
        )

        image_url = response["images"][0]["url"] if response else None

        if not image_url:
            logger.error(f"{self} error: image generation failed")
            yield ErrorFrame("Image generation failed")
            return

        logger.debug(f"Image generated at: {image_url}")

        # Load the image from the url
        logger.debug(f"Downloading image {image_url} ...")
        async with self._aiohttp_session.get(image_url) as response:
            logger.debug(f"Downloaded image {image_url}")
            encoded_image = await response.content.read()
            (image_bytes, size, format) = await asyncio.to_thread(load_image_bytes, encoded_image)

            frame = URLImageRawFrame(url=image_url, image=image_bytes, size=size, format=format)
            yield frame
