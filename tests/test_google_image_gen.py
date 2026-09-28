#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GoogleImageGenService response handling."""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types
from PIL import Image

from pipecat.frames.frames import ErrorFrame, URLImageRawFrame
from pipecat.services.google.image import GoogleImageGenService


def _png_bytes(mode: str, size: tuple[int, int] = (64, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "red").save(buffer, "PNG")
    return buffer.getvalue()


def _response(parts: list[types.Part]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
    )


def _image_part(mode: str = "RGB") -> types.Part:
    return types.Part(inline_data=types.Blob(data=_png_bytes(mode), mime_type="image/png"))


def _service(responses: list[types.GenerateContentResponse], **settings) -> GoogleImageGenService:
    service = GoogleImageGenService(
        api_key="test-key", settings=GoogleImageGenService.Settings(**settings)
    )
    service._client = MagicMock()
    service._client.aio.models.generate_content = AsyncMock(side_effect=responses)
    return service


async def _frames(service: GoogleImageGenService) -> list:
    return [frame async for frame in service.run_image_gen("a red square")]


@pytest.mark.asyncio
async def test_emits_inline_image():
    """The image part of a Gemini response becomes an RGB image frame."""
    service = _service([_response([types.Part(text="Here you go."), _image_part()])])

    frames = await _frames(service)

    assert len(frames) == 1
    frame = frames[0]
    assert isinstance(frame, URLImageRawFrame)
    assert frame.url is None
    assert frame.size == (64, 32)
    assert frame.format == "RGB"
    assert len(frame.image) == 64 * 32 * 3


@pytest.mark.asyncio
async def test_converts_alpha_images_to_rgb():
    """Images with an alpha channel are emitted as RGB."""
    service = _service([_response([_image_part("RGBA")])])

    frames = await _frames(service)

    assert frames[0].format == "RGB"
    assert len(frames[0].image) == 64 * 32 * 3


@pytest.mark.asyncio
async def test_requests_one_image_per_call():
    """number_of_images makes that many requests, each asking for an image."""
    service = _service([_response([_image_part()]) for _ in range(3)], number_of_images=3)

    frames = await _frames(service)

    assert len(frames) == 3
    generate = service._client.aio.models.generate_content
    assert generate.call_count == 3
    kwargs = generate.call_args.kwargs
    assert kwargs["model"] == "gemini-3.1-flash-image"
    assert kwargs["contents"] == "a red square"
    assert kwargs["config"].response_modalities == ["IMAGE"]
    assert kwargs["config"].image_config.aspect_ratio == "1:1"


@pytest.mark.asyncio
async def test_reports_error_when_response_has_no_image():
    """A response without an image part, such as a safety block, yields an error."""
    service = _service([_response([types.Part(text="I can't draw that.")])])

    frames = await _frames(service)

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)


@pytest.mark.asyncio
async def test_reports_api_errors():
    """An exception from the API is reported as an error frame."""
    service = _service([RuntimeError("boom")])

    frames = await _frames(service)

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "boom" in frames[0].error


def test_negative_prompt_warns():
    """A negative prompt is deprecated and ignored."""
    with pytest.warns(DeprecationWarning, match="negative_prompt"):
        GoogleImageGenService(
            api_key="test-key",
            settings=GoogleImageGenService.Settings(negative_prompt="blurry"),
        )
