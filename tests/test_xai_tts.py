#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for XAIHttpTTSService."""

import asyncio
import unittest

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    AggregatedTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.xai.tts import XAIHttpTTSService
from pipecat.tests.utils import run_test


@pytest.mark.asyncio
async def test_run_xai_tts_success(aiohttp_client):
    """xAI TTS should send the documented request body and emit PCM frames."""

    request_bodies = []

    async def handler(request):
        request_bodies.append(await request.json())

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "audio/pcm"},
        )
        await response.prepare(request)
        await response.write(b"\x00\x01\x02\x03" * 1024)
        await asyncio.sleep(0.01)
        await response.write(b"\x04\x05\x06\x07" * 1024)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/tts", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/v1/tts"))

    async with aiohttp.ClientSession() as session:
        tts_service = XAIHttpTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )

        down_frames, _ = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello from xAI.")],
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

    assert len(request_bodies) == 1
    assert request_bodies[0] == {
        "text": "Hello from xAI.",
        "voice_id": "eve",
        "language": "en",
        "output_format": {
            "codec": "pcm",
            "sample_rate": 24000,
        },
    }


if __name__ == "__main__":
    unittest.main()
