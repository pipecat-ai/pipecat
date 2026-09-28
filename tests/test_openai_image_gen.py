#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAIImageGenService response handling."""

import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types import ImagesResponse
from PIL import Image

from pipecat.frames.frames import ErrorFrame, URLImageRawFrame
from pipecat.services.openai.image import OpenAIImageGenService


def _png_bytes(mode: str, size: tuple[int, int] = (64, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "red").save(buffer, "PNG")
    return buffer.getvalue()


def _service(data: list[dict]) -> OpenAIImageGenService:
    service = OpenAIImageGenService(api_key="test-key", aiohttp_session=MagicMock())
    service._client = MagicMock()
    service._client.images.generate = AsyncMock(
        return_value=ImagesResponse.model_validate({"created": 0, "data": data})
    )
    return service


async def _frames(service: OpenAIImageGenService) -> list:
    return [frame async for frame in service.run_image_gen("a red square")]


@pytest.mark.asyncio
async def test_decodes_inline_base64_image():
    """GPT Image models return the image as base64 with no URL."""
    b64 = base64.b64encode(_png_bytes("RGB")).decode()
    service = _service([{"b64_json": b64}])

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
    b64 = base64.b64encode(_png_bytes("RGBA")).decode()
    service = _service([{"b64_json": b64}])

    frames = await _frames(service)

    assert frames[0].format == "RGB"
    assert len(frames[0].image) == 64 * 32 * 3


@pytest.mark.asyncio
async def test_downloads_hosted_url_image():
    """A response that carries a URL is fetched from that URL."""
    url = "https://example.com/image.png"
    service = _service([{"url": url}])
    response = MagicMock()
    response.content.read = AsyncMock(return_value=_png_bytes("RGB"))
    service._aiohttp_session.get.return_value.__aenter__ = AsyncMock(return_value=response)
    service._aiohttp_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    frames = await _frames(service)

    service._aiohttp_session.get.assert_called_once_with(url)
    assert frames[0].url == url
    assert frames[0].size == (64, 32)


@pytest.mark.asyncio
async def test_reports_error_when_response_has_no_image():
    """A data entry with neither a URL nor base64 data yields an error."""
    service = _service([{}])

    frames = await _frames(service)

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)


@pytest.mark.asyncio
async def test_default_model_is_sent_without_response_format():
    """GPT Image models reject response_format, so it is never sent."""
    b64 = base64.b64encode(_png_bytes("RGB")).decode()
    service = _service([{"b64_json": b64}])

    await _frames(service)

    kwargs = service._client.images.generate.call_args.kwargs
    assert kwargs["model"] == "gpt-image-2.5-flare"
    assert "response_format" not in kwargs
