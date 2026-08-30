#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GandrTTSService."""

import asyncio

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.gandr.tts import GandrTTSService
from pipecat.tests.utils import run_test


@pytest.mark.asyncio
async def test_run_gandr_tts_success(aiohttp_client):
    """Gandr TTS should send the documented request body and emit PCM frames."""

    request_bodies = []
    auth_headers = []
    pcm_audio = b"\x00\x01\x02\x03" * 1024

    async def handler(request):
        request_bodies.append(await request.json())
        auth_headers.append(request.headers.get("Authorization"))

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "audio/pcm"},
        )
        await response.prepare(request)
        # Split mid-sample to check that emitted frames stay sample-aligned.
        await response.write(pcm_audio[:2047])
        await asyncio.sleep(0.01)
        await response.write(pcm_audio[2047:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/v1/audio/speech"))

    async with aiohttp.ClientSession() as session:
        tts_service = GandrTTSService(
            api_key="gnd_test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )

        down_frames, _ = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello from Gandr.")],
        )

    frame_types = [type(frame) for frame in down_frames]
    assert AggregatedTextFrame in frame_types
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert TTSTextFrame in frame_types

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(frame.num_channels == 1 for frame in audio_frames)
    assert all(len(frame.audio) % 2 == 0 for frame in audio_frames)
    assert b"".join(frame.audio for frame in audio_frames) == pcm_audio

    assert len(request_bodies) == 1
    assert request_bodies[0] == {
        "model": "gandr-1",
        "input": "Hello from Gandr.",
        "voice": "gandr-mia",
        "response_format": "pcm",
    }
    assert auth_headers == ["Bearer gnd_test-key"]


@pytest.mark.asyncio
async def test_run_gandr_tts_http_error(aiohttp_client):
    """Gandr TTS should emit an ErrorFrame when the endpoint returns an error."""

    async def handler(request):
        await request.json()
        return web.Response(status=500, text="server error")

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/v1/audio/speech"))

    async with aiohttp.ClientSession() as session:
        tts_service = GandrTTSService(
            api_key="gnd_test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )

        expected_up_frames = [ErrorFrame]

        _, up_frames = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello from Gandr.", append_to_context=False)],
            expected_up_frames=expected_up_frames,
        )

    assert isinstance(up_frames[0], ErrorFrame)
    assert "status: 500" in up_frames[0].error
    assert "server error" in up_frames[0].error
