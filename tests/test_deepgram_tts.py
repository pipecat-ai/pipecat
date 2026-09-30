#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Deepgram TTS services."""

import asyncio

import aiohttp
import pytest
from aiohttp import web
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.task import PipelineParams
from pipecat.services.deepgram.tts import DeepgramHttpTTSService, DeepgramTTSService
from pipecat.tests.utils import run_test


def _websocket_rejection(status_code: int) -> InvalidStatus:
    """Build the exception `websockets` raises when a handshake is rejected."""
    return InvalidStatus(Response(status_code, "", Headers()))


@pytest.mark.asyncio
async def test_deepgram_rejected_api_key_makes_the_service_unusable(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(401)

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = DeepgramTTSService(api_key="wrong-key", sample_rate=24000)

    await service._connect_websocket()

    assert not service.is_usable


@pytest.mark.asyncio
async def test_deepgram_server_error_leaves_the_service_usable(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(503)

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = DeepgramTTSService(api_key="test-key", sample_rate=24000)

    await service._connect_websocket()

    assert service.is_usable


@pytest.mark.asyncio
async def test_deepgram_http_frames_stay_sample_aligned(aiohttp_client):
    """HTTP TTS audio frames hold whole 16-bit samples however the stream is split."""
    pcm_audio = b"\x00\x01\x02\x03" * 1024

    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "audio/l16"})
        await response.prepare(request)
        await response.write(pcm_audio[:2047])
        await asyncio.sleep(0.01)
        await response.write(pcm_audio[2047:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/speak", handler)
    client = await aiohttp_client(app)

    async with aiohttp.ClientSession() as session:
        tts_service = DeepgramHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("")).rstrip("/"),
            sample_rate=24000,
        )

        # Metrics on: the TTFA scan reads the audio as 16-bit samples.
        down_frames, _ = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello from Deepgram.")],
            pipeline_params=PipelineParams(enable_metrics=True),
        )

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
    assert all(len(frame.audio) % 2 == 0 for frame in audio_frames)
    assert b"".join(frame.audio for frame in audio_frames) == pcm_audio
