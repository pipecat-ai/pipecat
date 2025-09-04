#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI image generation service implementation.

This module provides integration with OpenAI's DALL-E image generation API
for creating images from text prompts.
"""

import io
from typing import AsyncGenerator, Literal, Optional

import aiohttp
from loguru import logger
from openai import AsyncOpenAI
from PIL import Image

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    URLImageRawFrame,
)
from pipecat.services.image_service import ImageGenService


class OpenAIImageGenService(ImageGenService):
    """OpenAI DALL-E image generation service.

    Provides image generation capabilities using OpenAI's DALL-E models.
    Supports various image sizes and can generate images from text prompts
    with configurable quality and style parameters.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        aiohttp_session: aiohttp.ClientSession,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        model: str = "dall-e-3",
    ):
        """Initialize the OpenAI image generation service.

        Args:
            api_key: OpenAI API key for authentication.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            aiohttp_session: HTTP session for downloading generated images.
            image_size: Target size for generated images.
            model: DALL-E model to use for generation. Defaults to "dall-e-3".
        """
        super().__init__()
        self.set_model_name(model)
        self._image_size = image_size
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate an image from a text prompt using OpenAI's DALL-E.

        Args:
            prompt: Text description of the image to generate.

        Yields:
            Frame: URLImageRawFrame containing the generated image data.
        """
        logger.debug(f"Generating image from prompt: {prompt}")

        image = await self._client.images.generate(
            prompt=prompt, model=self.model_name, n=1, size=self._image_size
        )

        image_url = image.data[0].url

        if not image_url:
            logger.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            frame = URLImageRawFrame(
                image=image.tobytes(),
                size=image.size,
                format=image.format,
                url=image_url,
            )
            yield frame
