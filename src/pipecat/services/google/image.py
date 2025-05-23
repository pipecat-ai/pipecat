#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
    class InputParams(BaseModel):
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
        super().__init__(**kwargs)
        self._params = params or GoogleImageGenService.InputParams()
        self._client = genai.Client(api_key=api_key)
        self.set_model_name(self._params.model)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate images from a text prompt using Google's Imagen model.

        Args:
            prompt (str): The text description to generate images from.

        Yields:
            Frame: Generated image frames or error frames.
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
