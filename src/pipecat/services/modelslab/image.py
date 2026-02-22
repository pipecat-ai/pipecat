#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ModelsLab image generation service implementation.

This module provides integration with ModelsLab's image generation API
for creating images from text prompts using Flux, SDXL, and hundreds of
community models.

API docs: https://docs.modelslab.com/image-generation/community-models/text2img
"""

import asyncio
import io
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.image_service import ImageGenService

MODELSLAB_API_URL = "https://modelslab.com/api/v6/images/text2img"
MODELSLAB_FETCH_URL = "https://modelslab.com/api/v6/images/fetch/{id}"
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 40  # 40 × 5s = 200s max


class ModelsLabImageGenService(ImageGenService):
    """ModelsLab image generation service.

    Provides text-to-image generation using ModelsLab's API. Supports
    Flux, Juggernaut XL, RealVisXL, DreamShaper XL, Stable Diffusion XL,
    and hundreds of community models.

    Handles ModelsLab's async processing pattern automatically: when the API
    returns ``status: "processing"``, the service polls the fetch endpoint
    until the image is ready.

    Usage::

        image_service = ModelsLabImageGenService(
            api_key=os.getenv("MODELSLAB_API_KEY"),
            aiohttp_session=session,
            model="flux",
        )
    """

    class InputParams(BaseModel):
        """Input parameters for ModelsLab image generation.

        Parameters:
            width: Image width in pixels. Defaults to 1024.
            height: Image height in pixels. Defaults to 1024.
            negative_prompt: Description of what to exclude from the image.
            safety_checker: Whether to enable content safety filtering. Defaults to False.
            enhance_prompt: Whether to automatically enhance the prompt. Defaults to False.
        """

        width: int = 1024
        height: int = 1024
        negative_prompt: Optional[str] = None
        safety_checker: bool = False
        enhance_prompt: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "flux",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the ModelsLabImageGenService.

        Args:
            api_key: ModelsLab API key. Obtain from https://modelslab.com/dashboard.
            aiohttp_session: HTTP client session for API calls and image downloads.
            model: ModelsLab model ID to use. Defaults to ``"flux"`` (Flux Schnell).
                Other options: ``"juggernaut-xl-v10"``, ``"realvisxlV50_v50Bakedvae"``,
                ``"dreamshaperXL10_alpha2Xl10"``, ``"sdxl"``, or any community model ID.
            params: Optional image generation parameters (dimensions, negative prompt, etc.).
            **kwargs: Additional arguments passed to parent ImageGenService.
        """
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._api_key = api_key
        self._aiohttp_session = aiohttp_session
        self._params = params or self.InputParams()

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate an image from a text prompt.

        Submits a generation request to the ModelsLab API and waits for
        completion, polling if the generation is asynchronous.

        Args:
            prompt: The text prompt describing the image to generate.

        Yields:
            URLImageRawFrame: Frame containing the generated image data and URL.
            ErrorFrame: If image generation fails or times out.
        """
        logger.debug(f"Generating image from prompt: {prompt}")

        # Build request payload
        payload = {
            "key": self._api_key,
            "model_id": self.model_name,
            "prompt": prompt,
            "width": self._params.width,
            "height": self._params.height,
            "samples": 1,
            "safety_checker": self._params.safety_checker,
            "enhance_prompt": self._params.enhance_prompt,
        }
        if self._params.negative_prompt:
            payload["negative_prompt"] = self._params.negative_prompt

        # Submit generation request
        async with self._aiohttp_session.post(
            MODELSLAB_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if not resp.ok:
                error_text = await resp.text()
                logger.error(f"ModelsLab API error {resp.status}: {error_text}")
                yield ErrorFrame(f"ModelsLab API error: {resp.status}")
                return
            data = await resp.json()

        image_url = await self._resolve_image_url(data)
        if not image_url:
            yield ErrorFrame("ModelsLab image generation failed or timed out")
            return

        logger.debug(f"Image generated at: {image_url}")

        # Download and decode the image
        async with self._aiohttp_session.get(image_url) as resp:
            if not resp.ok:
                logger.error(f"Failed to download image: {resp.status}")
                yield ErrorFrame("Failed to download generated image")
                return
            encoded_image = await resp.read()

        image_bytes, size, fmt = await asyncio.to_thread(self._decode_image, encoded_image)
        yield URLImageRawFrame(url=image_url, image=image_bytes, size=size, format=fmt)

    async def _resolve_image_url(self, data: dict) -> Optional[str]:
        """Resolve the image URL from an API response, polling if necessary.

        Args:
            data: The initial API response dictionary.

        Returns:
            The image URL on success, or ``None`` if generation failed/timed out.
        """
        status = data.get("status")

        # Immediate success
        if status == "success" and data.get("output"):
            output = data["output"]
            return output[0] if isinstance(output, list) else output

        # Async processing — poll fetch endpoint
        if status == "processing" and data.get("id"):
            job_id = data["id"]
            fetch_url = MODELSLAB_FETCH_URL.format(id=job_id)
            logger.debug(f"ModelsLab image generation queued, job_id={job_id}")

            for attempt in range(MAX_POLL_ATTEMPTS):
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                async with self._aiohttp_session.post(
                    fetch_url,
                    json={"key": self._api_key},
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if not resp.ok:
                        logger.error(f"ModelsLab fetch error {resp.status}")
                        return None
                    fetch_data = await resp.json()

                fetch_status = fetch_data.get("status")
                logger.debug(f"Poll {attempt + 1}/{MAX_POLL_ATTEMPTS}: status={fetch_status}")

                if fetch_status == "success" and fetch_data.get("output"):
                    output = fetch_data["output"]
                    return output[0] if isinstance(output, list) else output

                if fetch_status in ("error", "failed"):
                    logger.error(f"ModelsLab generation failed: {fetch_data.get('message')}")
                    return None

            logger.error("ModelsLab image generation timed out")
            return None

        # Error or unexpected response
        logger.error(f"ModelsLab unexpected response: status={status}, message={data.get('message')}")
        return None

    @staticmethod
    def _decode_image(encoded_image: bytes):
        """Decode raw image bytes into PIL format.

        Args:
            encoded_image: Raw image bytes downloaded from the URL.

        Returns:
            Tuple of (raw_bytes, size, format) suitable for URLImageRawFrame.
        """
        buffer = io.BytesIO(encoded_image)
        image = Image.open(buffer)
        return image.tobytes(), image.size, image.mode
