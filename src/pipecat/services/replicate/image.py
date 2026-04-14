#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Replicate image generation service implementation.

This module provides integration with Replicate-hosted image generation models
for creating images from text prompts.
"""

import asyncio
import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.image_service import ImageGenService
from pipecat.services.settings import NOT_GIVEN, ImageGenSettings, _NotGiven


@dataclass
class ReplicateImageGenSettings(ImageGenSettings):
    """Settings for the Replicate image generation service.

    Parameters:
        model: Replicate model identifier. Use ``owner/name`` for official
            models or ``owner/name:version`` for versioned community models.
        aspect_ratio: Aspect ratio for generated images.
        num_outputs: Number of images to generate.
        num_inference_steps: Number of denoising steps for the model.
        seed: Random seed for reproducible generation. ``None`` uses a random seed.
        output_format: Image format requested from the model.
        output_quality: Output quality value supported by the model.
        disable_safety_checker: Whether to disable the model safety checker.
        go_fast: Whether to use the model's faster generation mode.
        megapixels: Approximate megapixel count for generated images.
    """

    aspect_ratio: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    num_outputs: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    num_inference_steps: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    seed: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    output_format: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    output_quality: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    disable_safety_checker: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    go_fast: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    megapixels: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    def to_api_input(self) -> Dict[str, Any]:
        """Build the Replicate input payload from settings."""
        payload: Dict[str, Any] = {
            "aspect_ratio": self.aspect_ratio,
            "num_outputs": self.num_outputs,
            "num_inference_steps": self.num_inference_steps,
            "output_format": self.output_format,
            "output_quality": self.output_quality,
            "disable_safety_checker": self.disable_safety_checker,
            "go_fast": self.go_fast,
            "megapixels": self.megapixels,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        payload.update(self.extra)
        return payload


class ReplicateImageGenService(ImageGenService):
    """Replicate image generation service.

    Provides text-to-image generation using Replicate-hosted models. Official
    models use an ``owner/name`` identifier. Versioned community models can be
    addressed with ``owner/name:version``.
    """

    Settings = ReplicateImageGenSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for Replicate image generation.

        .. deprecated:: 0.0.105
            Use ``settings=ReplicateImageGenService.Settings(...)`` instead.

        Parameters:
            aspect_ratio: Aspect ratio for generated images. Defaults to ``"1:1"``.
            num_outputs: Number of images to generate. Defaults to ``1``.
            num_inference_steps: Number of denoising steps. Defaults to ``4``.
            seed: Random seed for reproducible generation. Defaults to ``None``.
            output_format: Output image format. Defaults to ``"webp"``.
            output_quality: Output quality value. Defaults to ``80``.
            disable_safety_checker: Whether to disable the safety checker. Defaults to ``False``.
            go_fast: Whether to use the fast model path. Defaults to ``True``.
            megapixels: Approximate megapixel count. Defaults to ``"1"``.
        """

        aspect_ratio: str = "1:1"
        num_outputs: int = Field(default=1, ge=1, le=4)
        num_inference_steps: int = Field(default=4, ge=1)
        seed: Optional[int] = None
        output_format: str = "webp"
        output_quality: int = Field(default=80, ge=0, le=100)
        disable_safety_checker: bool = False
        go_fast: bool = True
        megapixels: str = "1"

    _TERMINAL_ERROR_STATUSES = {"failed", "canceled", "cancelled"}

    def __init__(
        self,
        *,
        params: Optional[InputParams] = None,
        aiohttp_session: aiohttp.ClientSession,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        settings: Optional[Settings] = None,
        base_url: str = "https://api.replicate.com/v1",
        wait_timeout_secs: int = 60,
        poll_interval_secs: float = 0.5,
        max_poll_attempts: int = 120,
        **kwargs,
    ):
        """Initialize the ReplicateImageGenService.

        Args:
            params: Input parameters for image generation configuration.

                .. deprecated:: 0.0.105
                    Use ``settings=ReplicateImageGenService.Settings(...)`` instead.

            aiohttp_session: HTTP client session for Replicate requests and image downloads.
            api_token: Optional Replicate API token. If provided, sets the
                ``REPLICATE_API_TOKEN`` environment variable.
            model: Replicate model identifier. Defaults to
                ``"black-forest-labs/flux-schnell"``.

                .. deprecated:: 0.0.105
                    Use ``settings=ReplicateImageGenService.Settings(model=...)`` instead.

            settings: Runtime-configurable generation settings. When provided
                alongside deprecated parameters, ``settings`` values take precedence.
            base_url: Base URL for the Replicate HTTP API.
            wait_timeout_secs: Sync wait duration passed in the ``Prefer`` header.
            poll_interval_secs: Poll interval used when the initial sync request
                returns before output is available.
            max_poll_attempts: Maximum number of follow-up prediction polls.
            **kwargs: Additional arguments passed to parent ImageGenService.
        """
        default_settings = self.Settings(
            model="black-forest-labs/flux-schnell",
            aspect_ratio="1:1",
            num_outputs=1,
            num_inference_steps=4,
            seed=None,
            output_format="webp",
            output_quality=80,
            disable_safety_checker=False,
            go_fast=True,
            megapixels="1",
        )

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.aspect_ratio = params.aspect_ratio
                default_settings.num_outputs = params.num_outputs
                default_settings.num_inference_steps = params.num_inference_steps
                default_settings.seed = params.seed
                default_settings.output_format = params.output_format
                default_settings.output_quality = params.output_quality
                default_settings.disable_safety_checker = params.disable_safety_checker
                default_settings.go_fast = params.go_fast
                default_settings.megapixels = params.megapixels

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)
        self._aiohttp_session = aiohttp_session
        self._api_token = api_token or os.getenv("REPLICATE_API_TOKEN", "")
        self._base_url = base_url.rstrip("/")
        self._wait_timeout_secs = wait_timeout_secs
        self._poll_interval_secs = poll_interval_secs
        self._max_poll_attempts = max_poll_attempts
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token

    def _prediction_request(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build the Replicate prediction URL, headers, and request body."""
        model = self._settings.model or ""
        if "/" not in model:
            raise ValueError(
                "Replicate model must use 'owner/name' or 'owner/name:version' format"
            )

        input_payload = {"prompt": prompt, **self._settings.to_api_input()}
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
            "Prefer": f"wait={self._wait_timeout_secs}",
        }

        if ":" in model:
            _, version = model.rsplit(":", maxsplit=1)
            if not version:
                raise ValueError(
                    "Versioned Replicate models must use 'owner/name:version' format"
                )
            return (f"{self._base_url}/predictions", headers, {"version": version, "input": input_payload})

        owner, name = model.split("/", maxsplit=1)
        return (f"{self._base_url}/models/{owner}/{name}/predictions", headers, {"input": input_payload})

    def _extract_output_urls(self, prediction: Dict[str, Any]) -> list[str]:
        """Extract output URLs from a Replicate prediction response."""
        output = prediction.get("output")
        if output is None:
            return []
        if isinstance(output, str):
            return [output]
        if isinstance(output, dict):
            url = output.get("url")
            return [url] if isinstance(url, str) else []

        urls: list[str] = []
        if isinstance(output, list):
            for item in output:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict) and isinstance(item.get("url"), str):
                    urls.append(item["url"])
        return urls

    async def _poll_prediction(
        self,
        prediction: Dict[str, Any],
        headers: Dict[str, str],
        *,
        fallback_url: str | None,
    ) -> Dict[str, Any]:
        """Poll the prediction endpoint until output is available or it fails."""
        output_urls = self._extract_output_urls(prediction)
        if output_urls:
            return prediction

        prediction_url = prediction.get("urls", {}).get("get") or fallback_url

        for _ in range(self._max_poll_attempts):
            status = str(prediction.get("status", "")).lower()
            if status in self._TERMINAL_ERROR_STATUSES:
                error = prediction.get("error") or "prediction failed"
                raise RuntimeError(f"Replicate prediction failed: {error}")

            if not prediction_url:
                return prediction

            await asyncio.sleep(self._poll_interval_secs)
            async with self._aiohttp_session.get(prediction_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Replicate polling error ({response.status}): {error_text}"
                    )
                prediction = await response.json()
                output_urls = self._extract_output_urls(prediction)
                if output_urls:
                    return prediction

        raise TimeoutError("Replicate image generation timed out while waiting for output")

    @staticmethod
    def _load_image_frame(image_url: str | None, encoded_image: bytes) -> URLImageRawFrame:
        """Decode image bytes and build a Pipecat image frame."""
        image = Image.open(io.BytesIO(encoded_image))
        return URLImageRawFrame(
            url=image_url,
            image=image.tobytes(),
            size=image.size,
            format=image.format,
        )

    @staticmethod
    def _decode_data_url(data_url: str) -> bytes:
        """Decode a ``data:`` URL into raw bytes."""
        _, _, data = data_url.partition(",")
        if not data:
            raise ValueError("Replicate returned an invalid data URL")
        return base64.b64decode(data)

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate images from a text prompt using Replicate.

        Args:
            prompt: The text prompt to generate images from.

        Yields:
            URLImageRawFrame: Frame containing generated image data and metadata.
            ErrorFrame: If image generation fails.
        """
        logger.debug(f"Generating image from prompt with Replicate: {prompt}")
        await self.start_ttfb_metrics()

        try:
            url, headers, payload = self._prediction_request(prompt)
            async with self._aiohttp_session.post(url, json=payload, headers=headers) as response:
                if response.status not in {200, 201}:
                    error_text = await response.text()
                    await self.stop_ttfb_metrics()
                    yield ErrorFrame(
                        error=f"Replicate API error ({response.status}): {error_text}"
                    )
                    return
                prediction = await response.json()
                fallback_url = response.headers.get("Location")

            prediction = await self._poll_prediction(
                prediction,
                headers={"Authorization": headers["Authorization"]},
                fallback_url=fallback_url,
            )

            output_urls = self._extract_output_urls(prediction)
            if not output_urls:
                await self.stop_ttfb_metrics()
                yield ErrorFrame("Replicate image generation failed: no output returned")
                return

            await self.stop_ttfb_metrics()

            for image_url in output_urls:
                if image_url.startswith("data:"):
                    encoded_image = self._decode_data_url(image_url)
                else:
                    async with self._aiohttp_session.get(image_url) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            yield ErrorFrame(
                                error=f"Replicate image download error ({response.status}): {error_text}"
                            )
                            continue
                        encoded_image = await response.read()

                frame = await asyncio.to_thread(self._load_image_frame, image_url, encoded_image)
                yield frame

        except Exception as e:
            yield ErrorFrame(f"Replicate image generation error: {e}")
