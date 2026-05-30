#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for QwenTTSService."""

import asyncio
import base64
import json
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
from pipecat.services.qwen.tts import QwenTTSService
from pipecat.tests.utils import run_test


def _make_sse_chunk(data: dict) -> bytes:
    """Encode a dict as a DashScope SSE ``data:`` line."""
    return f"data:{json.dumps(data)}\n\n".encode()


@pytest.mark.asyncio
async def test_qwen_tts_success(aiohttp_client):
    """QwenTTSService should send the correct request and emit PCM audio frames."""

    AUDIO_BYTES = b"\x10\x20\x30\x40" * 512
    captured_requests: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        captured_requests.append(
            {
                "headers": dict(request.headers),
                "body": await request.json(),
            }
        )

        audio_b64 = base64.b64encode(AUDIO_BYTES).decode("ascii")

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)

        # First chunk: audio data
        await response.write(
            _make_sse_chunk(
                {
                    "output": {
                        "audio": {"data": audio_b64, "frame_id": 1},
                        "finish_reason": "null",
                    },
                    "usage": {"input_characters": 15, "output_characters": 15},
                    "request_id": "test-request-id",
                }
            )
        )
        await asyncio.sleep(0.01)

        # Final chunk: finish signal
        await response.write(
            _make_sse_chunk(
                {
                    "output": {"finish_reason": "stop"},
                    "usage": {"input_characters": 15, "output_characters": 15},
                    "request_id": "test-request-id",
                }
            )
        )
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/api/v1/services/aigc/text2audio/generation", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/api/v1/services/aigc/text2audio/generation"))

    async with aiohttp.ClientSession() as session:
        tts = QwenTTSService(
            api_key="test-dashscope-key",
            aiohttp_session=session,
            base_url=base_url,
            sample_rate=22050,
        )

        down_frames, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from Qwen TTS.")],
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types, "Expected TTSStartedFrame in output"
    assert TTSStoppedFrame in frame_types, "Expected TTSStoppedFrame in output"

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames, "Expected at least one TTSAudioRawFrame"
    assert all(f.sample_rate == 22050 for f in audio_frames)
    assert all(f.num_channels == 1 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == AUDIO_BYTES

    # Verify request format
    assert len(captured_requests) == 1
    req = captured_requests[0]

    assert req["headers"]["Authorization"] == "Bearer test-dashscope-key"
    assert req["headers"]["X-DashScope-SSE"] == "enable"

    body = req["body"]
    assert body["model"] == "qwen3-tts-flash"
    assert body["input"]["text"] == "Hello from Qwen TTS."
    assert body["parameters"]["voice"] == "Chelsie"
    assert body["parameters"]["format"] == "pcm"
    assert body["parameters"]["sample_rate"] == 22050


@pytest.mark.asyncio
async def test_qwen_tts_custom_settings(aiohttp_client):
    """QwenTTSService should use custom model/voice from Settings."""

    captured_requests: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        captured_requests.append(await request.json())
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(_make_sse_chunk({"output": {"finish_reason": "stop"}, "usage": {}}))
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/tts", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/tts"))

    async with aiohttp.ClientSession() as session:
        tts = QwenTTSService(
            api_key="key",
            aiohttp_session=session,
            base_url=base_url,
            settings=QwenTTSService.Settings(model="qwen-tts-latest", voice="Ethan"),
        )

        await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Custom voice test.")],
        )

    assert len(captured_requests) == 1
    body = captured_requests[0]
    assert body["model"] == "qwen-tts-latest"
    assert body["parameters"]["voice"] == "Ethan"


@pytest.mark.asyncio
async def test_qwen_tts_api_error(aiohttp_client):
    """QwenTTSService should emit an ErrorFrame on non-200 responses."""
    from pipecat.frames.frames import ErrorFrame

    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=401, text='{"error": "Invalid API key"}')

    app = web.Application()
    app.router.add_post("/tts", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/tts"))

    async with aiohttp.ClientSession() as session:
        tts = QwenTTSService(
            api_key="bad-key",
            aiohttp_session=session,
            base_url=base_url,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="This should fail.")],
        )

    # ErrorFrame is pushed upstream by the service
    all_frames = down_frames + up_frames
    error_frames = [f for f in all_frames if isinstance(f, ErrorFrame)]
    assert error_frames, "Expected an ErrorFrame on API error"
    assert any("Qwen TTS API error" in f.error for f in error_frames)


if __name__ == "__main__":
    unittest.main()
