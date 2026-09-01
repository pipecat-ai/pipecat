#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI image generation service implementation.

This module provides integration with OpenAI's image generation API for
creating images from text prompts.
"""

import base64
import io
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Literal, cast

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
from pipecat.services.settings import ImageGenSettings
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Hint set for the `size` argument to `images.generate`. The values mirror the
# Literal that `openai.resources.images.Images.generate` accepts on its `size`
# parameter (also visible as the `size` field of
# `openai.types.image_generate_params.ImageGenerateParams`). The OpenAI SDK
# does not export this as a named alias, so we redeclare it here.
#
# We cast `_settings.image_size` (a plain `str`) to this Literal at the API
# boundary so callers can still pass any size string (e.g. a newer value the
# SDK accepts before this list is updated). Invalid values surface as an
# OpenAI API error at runtime. Keep in sync on a best-effort basis when
# bumping the openai dep.
OpenAIImageSize = Literal[
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "256x256",
    "512x512",
    "1792x1024",
    "1024x1792",
]


@dataclass
class OpenAIImageGenSettings(ImageGenSettings):
    """Settings for the OpenAI image generation service.

    Parameters:
        model: Image generation model identifier.
        image_size: Target size for generated images.
    """

    image_size: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class OpenAIImageGenService(ImageGenService):
    """OpenAI image generation service.

    Provides image generation capabilities using OpenAI's GPT Image models.
    Supports various image sizes and can generate images from text prompts.
    """

    Settings = OpenAIImageGenSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
        | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ):
        """Initialize the OpenAI image generation service.

        Args:
            api_key: OpenAI API key for authentication.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            aiohttp_session: HTTP session for downloading generated images.
            image_size: Target size for generated images. Defaults to "1024x1024".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAIImageGenService.Settings(image_size=...)`` instead.
                    Will be removed in 2.0.0.

            model: Image generation model to use. Defaults to "gpt-image-2".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAIImageGenService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="gpt-image-2",
            image_size=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if image_size is not None:
            self._warn_init_param_moved_to_settings("image_size", "image_size")
            default_settings.image_size = image_size

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate an image from a text prompt using OpenAI's Images API.

        Args:
            prompt: Text description of the image to generate.

        Yields:
            Frame: URLImageRawFrame containing the generated image data.
        """
        logger.debug(f"Generating image from prompt: {prompt}")

        size = cast(OpenAIImageSize | None, assert_given(self._settings.image_size))
        image = await self._client.images.generate(
            prompt=prompt,
            model=assert_given(self._settings.model),
            n=1,
            size=size,
        )

        if not image.data:
            yield ErrorFrame("Image generation failed: no data returned")
            return

        # The GPT Image models always return the bytes inline; the response
        # carries a URL only when the model supports hosted results.
        image_url = image.data[0].url
        b64_data = image.data[0].b64_json

        if image_url:
            async with self._aiohttp_session.get(image_url) as response:
                image_bytes = await response.content.read()
        elif b64_data:
            image_bytes = base64.b64decode(b64_data)
        else:
            yield ErrorFrame("Image generation failed")
            return

        generated = Image.open(io.BytesIO(image_bytes))
        yield URLImageRawFrame(
            image=generated.tobytes(),
            size=generated.size,
            format=generated.mode,
            url=image_url,
        )
