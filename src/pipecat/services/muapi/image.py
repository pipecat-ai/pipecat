#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MuAPI image generation service implementation.

Provides integration with muapi.ai's unified API for generating images
from text prompts using 400+ models including Flux, Midjourney, GPT-4o,
Imagen, Seedream, HiDream, and more.
"""

import asyncio
import io
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger
from PIL import Image

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.image_service import ImageGenService
from pipecat.services.settings import NOT_GIVEN, ImageGenSettings, _NotGiven

BASE_URL = "https://api.muapi.ai/api/v1"

# Map of friendly model names to muapi endpoint paths
IMAGE_MODELS = {
    "flux-schnell": "flux-schnell",
    "flux-dev": "flux-dev",
    "flux-kontext-dev": "flux-kontext-dev",
    "flux-kontext-pro": "flux-kontext-pro",
    "flux-kontext-max": "flux-kontext-max",
    "hidream-fast": "hidream-fast",
    "hidream-dev": "hidream-dev",
    "hidream-full": "hidream-full",
    "midjourney": "midjourney",
    "gpt4o": "gpt4o",
    "gpt-image-2": "gpt-image-2",
    "imagen4": "imagen4",
    "imagen4-fast": "imagen4-fast",
    "seedream": "seedream",
    "reve": "reve",
    "ideogram": "ideogram",
    "hunyuan": "hunyuan",
    "wan2.1": "wan2.1",
    "qwen": "qwen",
}


@dataclass
class MuApiImageGenSettings(ImageGenSettings):
    """Settings for the MuAPI image generation service.

    Parameters:
        model: muapi.ai model identifier (e.g. 'flux-schnell', 'midjourney').
        width: Image width in pixels.
        height: Image height in pixels.
        num_inference_steps: Number of diffusion steps (model-dependent).
        poll_interval: Seconds between status polls (default 2).
        poll_timeout: Max seconds to wait for generation (default 300).
    """

    width: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    height: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    num_inference_steps: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    poll_interval: float = 2.0
    poll_timeout: float = 300.0

    def to_api_arguments(self) -> dict[str, Any]:
        """Build muapi payload extras, excluding NOT_GIVEN values."""
        args: dict[str, Any] = {}
        if self.width is not NOT_GIVEN:
            args["width"] = self.width
        if self.height is not NOT_GIVEN:
            args["height"] = self.height
        if self.num_inference_steps is not NOT_GIVEN:
            args["num_inference_steps"] = self.num_inference_steps
        return args


class MuApiImageGenService(ImageGenService):
    """MuAPI image generation service.

    Provides text-to-image generation using muapi.ai's unified API, which
    aggregates 400+ models including Flux, Midjourney, GPT-4o Image, Imagen,
    Seedream, HiDream, and more -- all through a single API key.

    The service follows muapi's async submit->poll pattern internally,
    presenting a synchronous interface to pipecat's frame pipeline.
    """

    Settings = MuApiImageGenSettings
    _settings: Settings

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        api_key: str | None = None,
        model: str | None = None,
        settings: MuApiImageGenSettings | None = None,
        **kwargs,
    ):
        """Initialize the MuApiImageGenService.

        Args:
            aiohttp_session: HTTP client session for API calls and image downloads.
            api_key: muapi.ai API key. Falls back to MUAPI_API_KEY env var.
            model: Model identifier (e.g. 'flux-schnell', 'midjourney').
                   Defaults to 'flux-schnell'.
            settings: Runtime-updatable settings object.
            **kwargs: Additional arguments passed to parent ImageGenService.
        """
        default_settings = self.Settings(
            model="flux-schnell",
            poll_interval=2.0,
            poll_timeout=300.0,
        )
        if model is not None:
            default_settings.model = model
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)
        self._aiohttp_session = aiohttp_session
        self._api_key = api_key or os.getenv("MUAPI_API_KEY", "")

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt to generate an image from.

        Yields:
            URLImageRawFrame: Frame containing the generated image data and metadata.
            ErrorFrame: If image generation fails.
        """
        model = self._settings.model
        endpoint = IMAGE_MODELS.get(model, model)

        logger.debug(f"Generating image with muapi model '{model}' from prompt: {prompt}")

        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {"prompt": prompt, **self._settings.to_api_arguments()}

        # Submit generation request
        async with self._aiohttp_session.post(
            f"{BASE_URL}/{endpoint}",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                yield ErrorFrame(error=f"MuAPI submit error ({resp.status}): {error_text}")
                return
            submit_data = await resp.json()

        request_id = submit_data.get("request_id")
        if not request_id:
            yield ErrorFrame("MuAPI: no request_id in submit response")
            return

        logger.debug(f"MuAPI request submitted: {request_id}")

        # Poll until completed
        deadline = asyncio.get_event_loop().time() + self._settings.poll_timeout
        image_url: str | None = None

        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(self._settings.poll_interval)
            async with self._aiohttp_session.get(
                f"{BASE_URL}/predictions/{request_id}/result",
                headers={"x-api-key": self._api_key},
            ) as poll_resp:
                if poll_resp.status != 200:
                    error_text = await poll_resp.text()
                    yield ErrorFrame(f"MuAPI poll error ({poll_resp.status}): {error_text}")
                    return
                poll_data = await poll_resp.json()

            status = poll_data.get("status", "pending")
            if status == "completed":
                outputs = poll_data.get("outputs", [])
                if outputs:
                    image_url = outputs[0]
                break
            if status in ("failed", "cancelled"):
                yield ErrorFrame(f"MuAPI image generation {status}: {poll_data.get('error', '')}")
                return

        if not image_url:
            yield ErrorFrame("MuAPI: image generation timed out or returned no output")
            return

        logger.debug(f"MuAPI image generated: {image_url}")

        # Download and decode image
        async with self._aiohttp_session.get(image_url) as img_resp:
            encoded_image = await img_resp.content.read()

        def load_image_bytes(data: bytes):
            buf = io.BytesIO(data)
            img = Image.open(buf)
            return (img.tobytes(), img.size, img.mode)

        (image_bytes, size, fmt) = await asyncio.to_thread(load_image_bytes, encoded_image)
        yield URLImageRawFrame(url=image_url, image=image_bytes, size=size, format=fmt)
