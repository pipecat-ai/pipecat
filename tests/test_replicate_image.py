#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ReplicateImageGenService."""

import io

import aiohttp
import pytest
from aiohttp import web
from PIL import Image

from pipecat.frames.frames import ErrorFrame, TextFrame, URLImageRawFrame
from pipecat.services.replicate.image import ReplicateImageGenService
from pipecat.tests.utils import run_test


def _make_test_image_bytes(format: str = "PNG") -> bytes:
    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_run_replicate_image_success_official_model(aiohttp_client):
    """Official Replicate models should return image frames from sync predictions."""

    image_bytes = _make_test_image_bytes()

    async def prediction_handler(request):
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["Prefer"] == "wait=60"
        payload = await request.json()
        assert payload["input"]["prompt"] == "a red square"
        assert payload["input"]["aspect_ratio"] == "1:1"
        return web.json_response(
            {
                "status": "processing",
                "output": [str(request.url.with_path("/image.png"))],
                "urls": {"get": str(request.url.with_path("/prediction-status"))},
            }
        )

    async def image_handler(_request):
        return web.Response(body=image_bytes, content_type="image/png")

    app = web.Application()
    app.router.add_post("/v1/models/black-forest-labs/flux-schnell/predictions", prediction_handler)
    app.router.add_get("/image.png", image_handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/v1")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        image_gen = ReplicateImageGenService(
            aiohttp_session=session,
            api_token="test-token",
            base_url=base_url,
        )

        down_frames, up_frames = await run_test(image_gen, frames_to_send=[TextFrame("a red square")])

        assert not up_frames
        assert isinstance(down_frames[0], TextFrame)
        assert isinstance(down_frames[1], URLImageRawFrame)
        assert down_frames[1].size == (2, 2)
        assert down_frames[1].format == "PNG"


@pytest.mark.asyncio
async def test_run_replicate_image_success_versioned_model(aiohttp_client):
    """Versioned community models should use the generic predictions endpoint."""

    image_bytes = _make_test_image_bytes()
    version = "53d5d1586a229bd033e060941789bfb0c177cefd5ef638f34b3099658343a897"

    async def prediction_handler(request):
        payload = await request.json()
        assert payload["version"] == version
        assert payload["input"]["prompt"] == "a blue square"
        return web.json_response(
            {
                "status": "successful",
                "output": [str(request.url.with_path("/image-versioned.png"))],
            }
        )

    async def image_handler(_request):
        return web.Response(body=image_bytes, content_type="image/png")

    app = web.Application()
    app.router.add_post("/v1/predictions", prediction_handler)
    app.router.add_get("/image-versioned.png", image_handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/v1")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        image_gen = ReplicateImageGenService(
            aiohttp_session=session,
            api_token="test-token",
            base_url=base_url,
            settings=ReplicateImageGenService.Settings(
                model=f"black-forest-labs/flux-schnell:{version}",
                aspect_ratio="1:1",
                num_outputs=1,
                num_inference_steps=4,
                seed=None,
                output_format="webp",
                output_quality=80,
                disable_safety_checker=False,
                go_fast=True,
                megapixels="1",
            ),
        )

        down_frames, up_frames = await run_test(image_gen, frames_to_send=[TextFrame("a blue square")])

        assert not up_frames
        assert isinstance(down_frames[1], URLImageRawFrame)


@pytest.mark.asyncio
async def test_run_replicate_image_polls_when_sync_response_has_no_output(aiohttp_client):
    """The service should poll the prediction URL if sync mode returns early."""

    image_bytes = _make_test_image_bytes()
    poll_count = 0

    async def prediction_handler(request):
        return web.json_response(
            {
                "status": "processing",
                "output": None,
                "urls": {"get": str(request.url.with_path("/v1/predictions/test-id"))},
            }
        )

    async def prediction_status_handler(request):
        nonlocal poll_count
        poll_count += 1
        return web.json_response(
            {
                "status": "processing",
                "output": [str(request.url.with_path("/image-polled.png"))],
            }
        )

    async def image_handler(_request):
        return web.Response(body=image_bytes, content_type="image/png")

    app = web.Application()
    app.router.add_post("/v1/models/black-forest-labs/flux-schnell/predictions", prediction_handler)
    app.router.add_get("/v1/predictions/test-id", prediction_status_handler)
    app.router.add_get("/image-polled.png", image_handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/v1")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        image_gen = ReplicateImageGenService(
            aiohttp_session=session,
            api_token="test-token",
            base_url=base_url,
            poll_interval_secs=0.001,
            max_poll_attempts=2,
        )

        down_frames, up_frames = await run_test(image_gen, frames_to_send=[TextFrame("poll me")])

        assert not up_frames
        assert poll_count == 1
        assert isinstance(down_frames[1], URLImageRawFrame)


@pytest.mark.asyncio
async def test_run_replicate_image_error(aiohttp_client):
    """Non-success responses should propagate an ErrorFrame upstream."""

    async def prediction_handler(_request):
        return web.Response(status=401, text="unauthorized")

    app = web.Application()
    app.router.add_post("/v1/models/black-forest-labs/flux-schnell/predictions", prediction_handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/v1")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        image_gen = ReplicateImageGenService(
            aiohttp_session=session,
            api_token="bad-token",
            base_url=base_url,
        )

        down_frames, up_frames = await run_test(
            image_gen,
            frames_to_send=[TextFrame("this should fail")],
        )

        assert isinstance(down_frames[0], TextFrame)
        assert len(up_frames) == 1
        assert isinstance(up_frames[0], ErrorFrame)
        assert "401" in up_frames[0].error
