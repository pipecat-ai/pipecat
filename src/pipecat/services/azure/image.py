#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI image generation service implementation.

This module provides integration with Azure's OpenAI image generation API
using REST endpoints for creating images from text prompts.
"""

import asyncio
import io
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import aiohttp
from PIL import Image

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.image_service import ImageGenService
from pipecat.services.settings import NOT_GIVEN, ImageGenSettings, _NotGiven, _warn_deprecated_param


@dataclass
class AzureImageGenSettings(ImageGenSettings):
    """Settings for the Azure image generation service.

    Parameters:
        model: Azure image generation model identifier.
        image_size: Target size for generated images.
    """

    image_size: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AzureImageGenServiceREST(ImageGenService):
    """Azure OpenAI REST-based image generation service.

    Provides image generation using Azure's OpenAI service via REST API.
    Supports asynchronous image generation with polling for completion
    and automatic image download and processing.
    """

    Settings = AzureImageGenSettings
    _settings: AzureImageGenSettings

    def __init__(
        self,
        *,
        image_size: Optional[str] = None,
        api_key: str,
        endpoint: str,
        model: Optional[str] = None,
        aiohttp_session: aiohttp.ClientSession,
        api_version="2023-06-01-preview",
        settings: Optional[AzureImageGenSettings] = None,
    ):
        """Initialize the AzureImageGenServiceREST.

        Args:
            image_size: Size specification for generated images (e.g., "1024x1024").

                .. deprecated:: 0.0.105
                    Use ``settings=AzureImageGenSettings(image_size=...)`` instead.

            api_key: Azure OpenAI API key for authentication.
            endpoint: Azure OpenAI endpoint URL.
            model: The image generation model to use.

                .. deprecated:: 0.0.105
                    Use ``settings=AzureImageGenSettings(model=...)`` instead.

            aiohttp_session: Shared aiohttp session for HTTP requests.
            api_version: Azure API version string. Defaults to "2023-06-01-preview".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = AzureImageGenSettings(
            model=None,
            image_size=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", AzureImageGenSettings, "model")
            default_settings.model = model

        if image_size is not None:
            _warn_deprecated_param("image_size", AzureImageGenSettings, "image_size")
            default_settings.image_size = image_size

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings)

        self._api_key = api_key
        self._azure_endpoint = endpoint
        self._api_version = api_version
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate an image from a text prompt using Azure OpenAI.

        Args:
            prompt: The text prompt describing the desired image.

        Yields:
            URLImageRawFrame containing the generated image data, or
            ErrorFrame if generation fails.
        """
        url = f"{self._azure_endpoint}openai/images/generations:submit?api-version={self._api_version}"

        headers = {"api-key": self._api_key, "Content-Type": "application/json"}

        body = {
            "prompt": prompt,
            "n": 1,
        }

        if self._settings.image_size is not None:
            body["size"] = self._settings.image_size

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
                    yield ErrorFrame("Image generation timed out")
                    return

                await asyncio.sleep(1)

                response = await self._aiohttp_session.get(operation_location, headers=headers)

                json_response = await response.json()
                status = json_response["status"]

            image_url = json_response["result"]["data"][0]["url"] if json_response else None
            if not image_url:
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
