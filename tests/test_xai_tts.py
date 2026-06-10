#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for XAIHttpTTSService and XAITTSService."""

import asyncio
import base64
import json
import unittest
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
import websockets
from aiohttp import web
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    AggregatedTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.xai.tts import XAIHttpTTSService, XAITTSService
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


@pytest.mark.asyncio
async def test_run_xai_websocket_tts_success():
    """xAI WS TTS should send text.delta+text.done and emit frames from audio.delta+audio.done."""

    captured: dict = {
        "request_path": None,
        "auth_header": None,
        "messages": [],
    }

    audio_bytes = b"\x00\x01\x02\x03" * 1024

    async def handler(ws):
        request = ws.request
        captured["request_path"] = request.path
        captured["auth_header"] = request.headers.get("Authorization")

        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                if msg.get("type") == "text.done":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "audio.delta",
                                "delta": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        )
                    )
                    await ws.send(json.dumps({"type": "audio.done", "trace_id": "test-trace"}))
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        base_url = f"ws://{host}:{port}/v1/tts"

        tts_service = XAITTSService(
            api_key="test-key",
            base_url=base_url,
            sample_rate=24000,
        )

        down_frames, _ = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello from xAI."), _SleepAfterSpeak(0.3)],
        )

    frame_types = [type(frame) for frame in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSAudioRawFrame in frame_types
    assert TTSStoppedFrame in frame_types

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(frame.num_channels == 1 for frame in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == audio_bytes

    assert captured["auth_header"] == "Bearer test-key"
    parsed = urlparse(captured["request_path"])
    query = parse_qs(parsed.query)
    assert query["voice"] == ["eve"]
    assert query["language"] == ["en"]
    assert query["codec"] == ["pcm"]
    assert query["sample_rate"] == ["24000"]

    types_sent = [m.get("type") for m in captured["messages"]]
    assert "text.delta" in types_sent
    assert "text.done" in types_sent
    delta_msg = next(m for m in captured["messages"] if m.get("type") == "text.delta")
    assert delta_msg["delta"] == "Hello from xAI."


# Small helper imported lazily to avoid circular import in fixture-lite tests.
def _SleepAfterSpeak(duration: float):
    from pipecat.tests.utils import SleepFrame

    return SleepFrame(sleep=duration)


if __name__ == "__main__":
    unittest.main()
