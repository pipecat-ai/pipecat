#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
from collections.abc import AsyncGenerator
from functools import partial
from os import PathLike
from typing import IO, Literal, Mapping, Protocol, cast, overload

import aiohttp
from loguru import logger
from openai import AsyncOpenAI
from openai.types import ImageModel, ImagesResponse
from PIL import Image
from typing_extensions import NotRequired, Required, TypedDict, Unpack, deprecated

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    URLImageRawFrame,
)
from pipecat.services.image_service import ImageGenService

_FileContent = IO[bytes] | bytes | PathLike[str]
_FileTypes = (
    _FileContent
    | tuple[str | None, _FileContent]
    | tuple[str | None, _FileContent, str | None]
    | tuple[str | None, _FileContent, str | None, Mapping[str, str]]
)


# Python doesn't have a way to `Omit` TypedDict keys to form new types,
# which means we have to manually copy and modify the params definitions.
# These are the same classes as in OpenAI's SDK, except that they have
# no `prompt` nor `model` keys.
class ImageGenerateParams(TypedDict, total=False):
    """Params for OpenAI image generation API / Create image endpoint.

    Learn more about the parameters here:
    https://platform.openai.com/docs/api-reference/images/create
    """

    background: Literal["transparent", "opaque", "auto"] | None
    moderation: Literal["low", "auto"] | None
    output_compression: int | None
    output_format: Literal["png", "jpeg", "webp"] | None
    quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | None
    response_format: Literal["url", "b64_json"] | None
    size: (
        Literal[
            "auto",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "256x256",
            "512x512",
            "1792x1024",
            "1024x1792",
        ]
        | None
    )
    style: Literal["vivid", "natural"] | None
    user: str


class ImageEditParams(TypedDict, total=False):
    """Params for OpenAI image generation API / Create image edit endpoint.

    Learn more about the parameters here:
    https://platform.openai.com/docs/api-reference/images/create
    """

    image: Required[_FileTypes | list[_FileTypes]]
    mask: _FileTypes
    quality: Literal["standard", "low", "medium", "high", "auto"] | None
    response_format: Literal["url", "b64_json"] | None
    size: Literal["256x256", "512x512", "1024x1024"] | None
    user: str


ImageParams = ImageGenerateParams | ImageEditParams


class OpenAIImageGenService(ImageGenService):
    class ImageGenerateInitParams(TypedDict):
        modality: Literal["generate"]
        params: NotRequired[ImageGenerateParams]

    class ImageEditInitParams(TypedDict):
        modality: Literal["edit"]
        params: ImageEditParams

    ImageInitParams = ImageGenerateInitParams | ImageEditInitParams

    class ImageGenerationFunction(Protocol):
        async def __call__(self, *, prompt: str) -> ImagesResponse: ...

    @overload
    @deprecated('Use `params["image_size"]` to set the image size instead')
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        model: str = "dall-e-3",
    ): ...

    @overload
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession,
        model: ImageModel = "dall-e-3",
        **kwargs: Unpack[ImageGenerateInitParams],
    ): ...

    @overload
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession,
        model: ImageModel = "dall-e-3",
        **kwargs: Unpack[ImageEditInitParams],
    ): ...

    def __init__(
        self,
        *,
        api_key,
        base_url=None,
        aiohttp_session,
        image_size=None,
        model="dall-e-3",
        **kwargs,
    ):
        unpacked_args: OpenAIImageGenService.ImageInitParams | None = None
        is_deprecated_overload_call = image_size is not None and not kwargs
        if is_deprecated_overload_call:
            # Adapt the deprecated overload params into the standard approach
            unpacked_args = OpenAIImageGenService.ImageGenerateInitParams(
                {"modality": "generate", "params": ImageGenerateParams({"size": image_size})}
            )
        elif not kwargs:
            # This state shouldn't be reached, but since we can't easily make invalid
            # states unreachable in Python, we just raise an error instead
            raise ValueError("Incompatible parameters in function call")

        # At this point, we can consider that either `unpacked_args` or `kwargs`
        # contain the keys and values we expect from the type system
        # Important: Note that this is not runtime checked - a runtime error might occur
        # down the line if the user doesn't provide what the type checker asks for.
        unpacked_args = cast(OpenAIImageGenService.ImageInitParams, unpacked_args or kwargs)

        super().__init__()
        self.set_model_name(model)
        self._aiohttp_session = aiohttp_session

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._generate_image = self.init_generation_function(client, unpacked_args)

    def init_generation_function(
        self,
        client: AsyncOpenAI,
        unpacked_args: ImageInitParams,
    ) -> ImageGenerationFunction:
        """Initialize an image generation function using the OpenAI API based on the specified modality.

        This method creates a partial function for image generation or editing based on the 'modality'
        parameter.

        Raises:
            KeyError: If `unpacked_args` doesn't contain "modality" or "params"
            ValueError: If the provided modality is not one of the supported options
        """
        params = {
            "n": 1,
            "model": self.model_name,
        }
        match unpacked_args["modality"]:
            case "generate":
                generation_function = client.images.generate
                if "params" in unpacked_args:
                    params = {**unpacked_args["params"], **params}
            case "edit":
                generation_function = client.images.edit
                params = {**unpacked_args["params"], **params}
            case _:
                raise ValueError(f'Unknown modality "{unpacked_args["modality"]}"')

        return partial(generation_function, **params)

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating image from prompt: {prompt}")

        image = await self._generate_image(prompt=prompt)
        if not image.data or not image.data[0].url:
            logger.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        image_url = image.data[0].url

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            frame = URLImageRawFrame(image_url, image.tobytes(), image.size, image.format)
            yield frame
