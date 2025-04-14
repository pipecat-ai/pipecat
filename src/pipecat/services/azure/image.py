#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
from typing import AsyncGenerator

import aiohttp
from loguru import logger
from PIL import Image

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.image_service import ImageGenService


class AzureImageGenServiceREST(ImageGenService):
    def __init__(
        self,
        *,
        image_size: str,
        api_key: str,
        endpoint: str,
        model: str,
        aiohttp_session: aiohttp.ClientSession,
        api_version="2023-06-01-preview",
    ):
        super().__init__()

        self._api_key = api_key
        self._azure_endpoint = endpoint
        self._api_version = api_version
        self.set_model_name(model)
        self._image_size = image_size
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        url = f"{self._azure_endpoint}openai/images/generations:submit?api-version={self._api_version}"

        headers = {"api-key": self._api_key, "Content-Type": "application/json"}

        body = {
            # Enter your prompt text here
            "prompt": prompt,
            "size": self._image_size,
            "n": 1,
        }

        async with self._aiohttp_session.post(url, headers=headers, json=body) as submission:
            # We never get past this line, because this header isn't
            # defined on a 429 response, but something is eating our
            # exceptions!
            operation_location = submission.headers["operation-location"]
            status = ""
            attempts_left = 120
            json_response = None
            while status != "succeeded":
                attempts_left -= 1
                if attempts_left == 0:
                    logger.error(f"{self} error: image generation timed out")
                    yield ErrorFrame("Image generation timed out")
                    return

                await asyncio.sleep(1)

                response = await self._aiohttp_session.get(operation_location, headers=headers)

                json_response = await response.json()
                status = json_response["status"]

            image_url = json_response["result"]["data"][0]["url"] if json_response else None
            if not image_url:
                logger.error(f"{self} error: image generation failed")
                yield ErrorFrame("Image generation failed")
                return

            # Load the image from the url
            async with self._aiohttp_session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                frame = URLImageRawFrame(
                    url=image_url, image=image.tobytes(), size=image.size, format=image.format
                )
                yield frame
