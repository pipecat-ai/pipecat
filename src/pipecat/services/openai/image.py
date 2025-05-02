#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
from functools import partial
from sys import version_info
from typing import Any, AsyncGenerator, Literal, Mapping, Optional, overload

if version_info < (3, 13):
    from typing_extensions import deprecated
else:
    from warnings import deprecated

import aiohttp
from loguru import logger
from openai import AsyncOpenAI
from openai.types import ImageModel
from PIL import Image

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    URLImageRawFrame,
)
from pipecat.services.image_service import ImageGenService


class OpenAIImageGenService(ImageGenService):
    Modality = Literal["generate", "edit", "create_variation"]
    InputParams = Mapping[str, Any]

    @overload
    @deprecated('Use `params["image_size"]` to set the image size instead')
    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str],
        aiohttp_session: aiohttp.ClientSession,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        model: Optional[str],
    ): ...

    @overload
    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str],
        aiohttp_session: aiohttp.ClientSession,
        model: Optional[ImageModel],
        modality: Optional[Literal["generate"]],
        params: Optional[InputParams],
        **kwargs,
    ): ...

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        aiohttp_session: aiohttp.ClientSession,
        modality: Modality = "generate",
        image_size: Optional[str] = None,
        model: str = "dall-e-3",
        params: InputParams = dict(),
        **kwargs,
    ):
        # Fail fast if `params` contains any of `model` or `n` keys,
        # as it would override such params when instantiating
        # `self._generate_image` down below.
        # The value for `n` is forced to be `1` for now, since there's
        # no support for generating multiple images yet.
        # The value for `model` should be passed as a top-level method
        # argument instead.
        invalid_params = [param for param in ["model", "n"] if param in params]
        if invalid_params:
            raise ValueError(
                f"Protected parameter(s) {', '.join(invalid_params)} cannot be specified inside params."
            )

        # Temporary guard until the remaining modalities are implemented
        if modality != "generate":
            raise ValueError(f"Unimplemented or unknown modality '{modality}'")

        super().__init__(**kwargs)
        self.set_model_name(model)
        self._aiohttp_session = aiohttp_session

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._generate_image = partial(
            client.images.generate,
            model=self.model_name,
            n=1,
            size=image_size,
            **params,
        )

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating image from prompt: {prompt}")

        image = await self._generate_image(prompt=prompt)
        image_url = image.data[0].url

        if not image_url:
            logger.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            frame = URLImageRawFrame(image_url, image.tobytes(), image.size, image.format)
            yield frame
