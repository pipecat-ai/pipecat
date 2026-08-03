#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for BlandTTSService."""

import struct

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.bland.tts import BlandTTSService
from pipecat.tests.utils import run_test

DEFAULT_VOICE_ID = "f04af0e5-1a80-48a9-b02d-52f30d417cfa"
OTHER_VOICE_ID = "c18a1cd5-91ef-4b06-841a-e58b8b487e8c"


def _pcm_bytes(num_samples: int = 4096) -> bytes:
    """Bare little-endian int16 PCM, which is what ``container: raw`` returns."""
    return struct.pack(f"<{num_samples}h", *(((i * 97) % 2000) - 1000 for i in range(num_samples)))


async def _serve(handler):
    app = web.Application()
    app.router.add_post("/v2/tts", handler)
    return app


def _audio_of(frames) -> bytes:
    return b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame))


@pytest.mark.asyncio
async def test_run_bland_tts_success(aiohttp_client):
    """Sends the documented request and emits PCM frames from the response."""
    requests = []
    payload = _pcm_bytes()

    async def handler(request):
        requests.append((request.headers.get("Authorization"), await request.json()))
        return web.Response(body=payload, content_type="audio/pcm")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )
        down_frames, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from Bland.")],
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types

    auth, body = requests[0]
    assert auth == "Bearer test-key"
    assert body["text"] == "Hello from Bland."
    assert body["voice"] == DEFAULT_VOICE_ID
    # 24000 is a rate Bland renders directly, so it is requested as-is.
    assert body["audio"] == {"encoding": "pcm_s16le", "sample_rate": 24000}
    assert "controls" not in body
    # fields the request shape does not define
    assert "language" not in body
    assert "output_format" not in body
    assert "voice_id" not in body

    audio = _audio_of(down_frames)
    assert audio == payload
    assert not audio.startswith(b"RIFF")
    assert {f.sample_rate for f in down_frames if isinstance(f, TTSAudioRawFrame)} == {24000}


@pytest.mark.asyncio
async def test_bland_tts_resamples_unsupported_pipeline_rate(aiohttp_client):
    """A pipeline rate Bland cannot emit falls back to 48 kHz and is resampled down."""
    requests = []
    payload = _pcm_bytes(4800)

    async def handler(request):
        requests.append(await request.json())
        return web.Response(body=payload, content_type="audio/pcm")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=22050,
        )
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert requests[0]["audio"]["sample_rate"] == 48000
    audio = _audio_of(down_frames)
    assert audio
    assert audio != payload
    assert {f.sample_rate for f in down_frames if isinstance(f, TTSAudioRawFrame)} == {22050}


@pytest.mark.asyncio
async def test_bland_tts_reassembles_audio_split_across_chunks(aiohttp_client):
    """A split at an odd byte lands mid-sample; nothing may be dropped or reordered."""
    payload = _pcm_bytes()
    splits = [1, 3, 1000, 2001, len(payload)]

    async def handler(request):
        response = web.StreamResponse(headers={"content-type": "audio/pcm"})
        await response.prepare(request)
        start = 0
        for end in splits:
            await response.write(payload[start:end])
            start = end
        await response.write_eof()
        return response

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert _audio_of(down_frames) == payload


@pytest.mark.asyncio
async def test_bland_tts_settings_payload(aiohttp_client):
    """Settings map into the request body."""
    requests = []

    async def handler(request):
        requests.append(await request.json())
        return web.Response(body=_pcm_bytes(), content_type="audio/pcm")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
            settings=BlandTTSService.Settings(
                voice=OTHER_VOICE_ID, expressiveness=0.9, stability=0.4
            ),
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    body = requests[0]
    assert body["voice"] == OTHER_VOICE_ID
    assert body["controls"] == {"expressiveness": 0.9, "stability": 0.4}


@pytest.mark.asyncio
async def test_bland_tts_partial_controls(aiohttp_client):
    """Only controls the caller set are sent, so unset ones keep Bland's defaults."""
    requests = []

    async def handler(request):
        requests.append(await request.json())
        return web.Response(body=_pcm_bytes(), content_type="audio/pcm")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
            settings=BlandTTSService.Settings(stability=0.4),
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert requests[0]["controls"] == {"stability": 0.4}


@pytest.mark.asyncio
async def test_bland_tts_error_response(aiohttp_client):
    """A non-200 response yields an ErrorFrame carrying the v2 error code and message."""

    async def handler(request):
        return web.json_response(
            {"error": {"code": "voice_not_found", "message": "Voice was not found."}},
            status=404,
        )

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "voice_not_found" in errors[0].error
    assert "Voice was not found." in errors[0].error


@pytest.mark.asyncio
async def test_bland_tts_non_json_error_response(aiohttp_client):
    """A gateway error with an HTML body still surfaces as an ErrorFrame."""

    async def handler(request):
        return web.Response(body=b"<html>gateway</html>", status=502, content_type="text/html")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "502" in errors[0].error
