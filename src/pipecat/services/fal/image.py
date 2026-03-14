#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fal's image generation service implementation.

This module provides integration with Fal's image generation API
for creating images from text prompts using various AI models.
"""

import asyncio
import io
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional, Union

import aiohttp
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.image_service import ImageGenService
from pipecat.services.settings import NOT_GIVEN, ImageGenSettings, _NotGiven, _warn_deprecated_param


@dataclass
class FalImageGenSettings(ImageGenSettings):
    """Settings for the Fal image generation service.

    Parameters:
        model: Fal.ai model identifier.
        seed: Random seed for reproducible generation. ``None`` uses a random seed.
        num_inference_steps: Number of inference steps for generation.
        num_images: Number of images to generate.
        image_size: Image dimensions as a string preset or dict with width/height.
        expand_prompt: Whether to automatically expand/enhance the prompt.
        enable_safety_checker: Whether to enable content safety filtering.
        format: Output image format.
    """

    seed: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    num_inference_steps: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    num_images: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    image_size: str | Dict[str, int] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    expand_prompt: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_safety_checker: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    format: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    def to_api_arguments(self) -> Dict[str, Any]:
        """Build the Fal API arguments dict from settings, excluding None values."""
        args: Dict[str, Any] = {}
        if self.seed is not None:
            args["seed"] = self.seed
        args["num_inference_steps"] = self.num_inference_steps
        args["num_images"] = self.num_images
        args["image_size"] = self.image_size
        args["expand_prompt"] = self.expand_prompt
        args["enable_safety_checker"] = self.enable_safety_checker
        args["format"] = self.format
        return args


class FalImageGenService(ImageGenService):
    """Fal's image generation service.

    Provides text-to-image generation using Fal.ai's API with configurable
    parameters for image quality, safety, and format options.
    """

    Settings = FalImageGenSettings
    _settings: FalImageGenSettings

    class InputParams(BaseModel):
        """Input parameters for Fal.ai image generation.

        .. deprecated:: 0.0.105
            Use ``settings=FalImageGenSettings(...)`` instead.

        Parameters:
            seed: Random seed for reproducible generation. If None, uses random seed.
            num_inference_steps: Number of inference steps for generation. Defaults to 8.
            num_images: Number of images to generate. Defaults to 1.
            image_size: Image dimensions as string preset or dict with width/height. Defaults to "square_hd".
            expand_prompt: Whether to automatically expand/enhance the prompt. Defaults to False.
            enable_safety_checker: Whether to enable content safety filtering. Defaults to True.
            format: Output image format. Defaults to "png".
        """

        seed: Optional[int] = None
        num_inference_steps: int = 8
        num_images: int = 1
        image_size: Union[str, Dict[str, int]] = "square_hd"
        expand_prompt: bool = False
        enable_safety_checker: bool = True
        format: str = "png"

    _settings: FalImageGenSettings

    def __init__(
        self,
        *,
        params: Optional[InputParams] = None,
        aiohttp_session: aiohttp.ClientSession,
        model: Optional[str] = None,
        key: Optional[str] = None,
        settings: Optional[FalImageGenSettings] = None,
        **kwargs,
    ):
        """Initialize the FalImageGenService.

        Args:
            params: Input parameters for image generation configuration.

                .. deprecated:: 0.0.105
                    Use ``settings=FalImageGenSettings(...)`` instead.

            aiohttp_session: HTTP client session for downloading generated images.
            model: The Fal.ai model to use for generation. Defaults to "fal-ai/fast-sdxl".

                .. deprecated:: 0.0.105
                    Use ``settings=FalImageGenSettings(model=...)`` instead.

            key: Optional API key for Fal.ai. If provided, sets FAL_KEY environment variable.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to parent ImageGenService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = FalImageGenSettings(
            model="fal-ai/fast-sdxl",
            seed=None,
            num_inference_steps=8,
            num_images=1,
            image_size="square_hd",
            expand_prompt=False,
            enable_safety_checker=True,
            format="png",
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", FalImageGenSettings, "model")
            default_settings.model = model

        if params is not None:
            _warn_deprecated_param("params", FalImageGenSettings)
            if not settings:
                default_settings.seed = params.seed
                default_settings.num_inference_steps = params.num_inference_steps
                default_settings.num_images = params.num_images
                default_settings.image_size = params.image_size
                default_settings.expand_prompt = params.expand_prompt
                default_settings.enable_safety_checker = params.enable_safety_checker
                default_settings.format = params.format

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)
        self._aiohttp_session = aiohttp_session
        self._api_key = key or os.getenv("FAL_KEY", "")
        if key:
            os.environ["FAL_KEY"] = key

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt to generate an image from.

        Yields:
            URLImageRawFrame: Frame containing the generated image data and metadata.
            ErrorFrame: If image generation fails.
        """

        def load_image_bytes(encoded_image: bytes):
            buffer = io.BytesIO(encoded_image)
            image = Image.open(buffer)
            return (image.tobytes(), image.size, image.format)

        logger.debug(f"Generating image from prompt: {prompt}")

        headers = {
            "Authorization": f"Key {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": prompt, **self._settings.to_api_arguments()}

        async with self._aiohttp_session.post(
            f"https://fal.run/{self._settings.model}",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                yield ErrorFrame(error=f"Fal API error ({resp.status}): {error_text}")
                return
            response = await resp.json()

        image_url = response["images"][0]["url"] if response else None

        if not image_url:
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
