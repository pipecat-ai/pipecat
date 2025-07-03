#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google AI image generation service implementation.

This module provides integration with Google's Imagen model for generating
images from text prompts using the Google AI API.
"""

import io
import os

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from typing import AsyncGenerator, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.image_service import ImageGenService

try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


class GoogleImageGenService(ImageGenService):
    """Google AI image generation service using Imagen models.

    Provides text-to-image generation capabilities using Google's Imagen models
    through the Google AI API. Supports multiple image generation and negative
    prompting for enhanced control over generated content.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Google image generation.

        Parameters:
            number_of_images: Number of images to generate (1-8). Defaults to 1.
            model: Google Imagen model to use. Defaults to "imagen-3.0-generate-002".
            negative_prompt: Optional negative prompt to guide what not to include.
        """

        number_of_images: int = Field(default=1, ge=1, le=8)
        model: str = Field(default="imagen-3.0-generate-002")
        negative_prompt: Optional[str] = Field(default=None)

    def __init__(
        self,
        *,
        api_key: str,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the GoogleImageGenService with API key and parameters.

        Args:
            api_key: Google AI API key for authentication.
            params: Configuration parameters for image generation. Defaults to InputParams().
            **kwargs: Additional arguments passed to the parent ImageGenService.
        """
        super().__init__(**kwargs)
        self._params = params or GoogleImageGenService.InputParams()
        self._client = genai.Client(api_key=api_key)
        self.set_model_name(self._params.model)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Google image generation service supports metrics.
        """
        return True

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate images from a text prompt using Google's Imagen model.

        Args:
            prompt: The text description to generate images from.

        Yields:
            Frame: Generated URLImageRawFrame objects containing the generated
                images, or ErrorFrame objects if generation fails.

        Raises:
            Exception: If there are issues with the Google AI API or image processing.
        """
        logger.debug(f"Generating image from prompt: {prompt}")
        await self.start_ttfb_metrics()

        try:
            response = await self._client.aio.models.generate_images(
                model=self._params.model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=self._params.number_of_images,
                    negative_prompt=self._params.negative_prompt,
                ),
            )
            await self.stop_ttfb_metrics()

            if not response or not response.generated_images:
                logger.error(f"{self} error: image generation failed")
                yield ErrorFrame("Image generation failed")
                return

            for img_response in response.generated_images:
                # Google returns the image data directly
                image_bytes = img_response.image.image_bytes
                image = Image.open(io.BytesIO(image_bytes))

                frame = URLImageRawFrame(
                    url=None,  # Google doesn't provide URLs, only image data
                    image=image.tobytes(),
                    size=image.size,
                    format=image.format,
                )
                yield frame

        except Exception as e:
            logger.error(f"{self} error generating image: {e}")
            yield ErrorFrame(f"Image generation error: {str(e)}")
