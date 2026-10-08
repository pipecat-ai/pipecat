#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google AI image generation service implementation.

This module provides integration with Google's Gemini image models for
generating images from text prompts using the Gemini API.
"""

import io
import os

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.google.utils import update_google_client_http_options
from pipecat.services.image_service import ImageGenService
from pipecat.services.settings import ImageGenSettings
from pipecat.utils.deprecation import deprecated, warn_deprecated
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

try:
    import google.genai as genai
    from google.genai import types
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Google AI, you need to `uv add "pipecat-ai[google]"`.')
    raise ImportError(f"Missing module: {e}") from e


@dataclass
class GoogleImageGenSettings(ImageGenSettings):
    """Settings for the Google image generation service.

    Parameters:
        model: Gemini image model identifier.
        number_of_images: Number of images to generate per prompt.
        aspect_ratio: Aspect ratio of generated images, such as ``"1:1"`` or
            ``"16:9"``. ``None`` leaves the model's own default in place.
        negative_prompt: Text describing what not to include in generated images.

            .. deprecated:: 1.13.0
                No replacement. Gemini image models do not accept a negative
                prompt, so the value is ignored. Will be removed in 2.0.0.
    """

    number_of_images: int | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    aspect_ratio: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    negative_prompt: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GoogleImageGenService(ImageGenService):
    """Google AI image generation service using Gemini image models.

    Provides text-to-image generation using Google's Gemini image models
    through the Gemini API. Each image is a separate request, so
    ``number_of_images`` images are generated one after another.
    """

    Settings = GoogleImageGenSettings
    _settings: Settings

    @deprecated(
        "`GoogleImageGenService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `GoogleImageGenService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for Google image generation.

        .. deprecated:: 0.0.105
            Use ``settings=GoogleImageGenService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            number_of_images: Number of images to generate (1-8). Defaults to 1.
            model: Gemini image model to use. Defaults to "gemini-3.1-flash-image".
            negative_prompt: Ignored; Gemini image models do not accept a
                negative prompt.
        """

        number_of_images: int = Field(default=1, ge=1, le=8)
        model: str = Field(default="gemini-3.1-flash-image")
        negative_prompt: str | None = Field(default=None)

    def __init__(
        self,
        *,
        api_key: str,
        params: InputParams | None = None,
        http_options: Any | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the GoogleImageGenService with API key and parameters.

        Args:
            api_key: Google AI API key for authentication.
            params: Configuration parameters for image generation.

                .. deprecated:: 0.0.105
                    Use ``settings=GoogleImageGenService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            http_options: HTTP options for the client.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent ImageGenService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="gemini-3.1-flash-image",
            number_of_images=1,
            aspect_ratio="1:1",
            negative_prompt=None,
        )

        # 2. Apply params overrides (deprecated)
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.model = params.model
                default_settings.number_of_images = params.number_of_images
                default_settings.negative_prompt = params.negative_prompt

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        if default_settings.negative_prompt:
            warn_deprecated(
                "`negative_prompt` is deprecated since 1.13.0 and will be removed in 2.0.0. "
                "No replacement. Gemini image models do not accept a negative prompt, "
                "so the value is ignored.",
                stacklevel=2,
            )

        super().__init__(settings=default_settings, **kwargs)

        # Add client header
        http_options = update_google_client_http_options(http_options)

        self._client = genai.Client(api_key=api_key, http_options=http_options)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Google image generation service supports metrics.
        """
        return True

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate images from a text prompt using a Gemini image model.

        Args:
            prompt: The text description to generate images from.

        Yields:
            Frame: Generated URLImageRawFrame objects containing the generated
                images, or ErrorFrame objects if generation fails.
        """
        logger.debug(f"Generating image from prompt: {prompt}")
        await self.start_ttfb_metrics()

        try:
            model = assert_given(self._settings.model)
            if model is None:
                yield ErrorFrame("Google image generation model must be specified")
                return

            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=assert_given(self._settings.aspect_ratio)
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            for _ in range(assert_given(self._settings.number_of_images)):
                response = await self._client.aio.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                await self.stop_ttfb_metrics()

                image_bytes = _get_image_bytes(response)
                if image_bytes is None:
                    yield ErrorFrame("Image generation failed: no image returned")
                    return

                # Output transports render RGB video by default.
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                yield URLImageRawFrame(
                    url=None,  # Gemini returns the image inline, not as a URL
                    image=image.tobytes(),
                    size=image.size,
                    format=image.mode,
                )

        except Exception as e:
            yield ErrorFrame(f"Image generation error: {str(e)}")


def _get_image_bytes(response: types.GenerateContentResponse) -> bytes | None:
    """Return the bytes of the first image part in a response, if any."""
    for candidate in response.candidates or []:
        if not candidate.content:
            continue
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data
    return None
