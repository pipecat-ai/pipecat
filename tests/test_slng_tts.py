#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Tests for SlngTTSService."""

import asyncio
import base64
import json

import pytest
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.tests.utils import SleepFrame, run_test


@pytest.mark.asyncio
async def test_slng_tts_ws_audio_chunk_and_flushed():
    """WS TTS should emit TTSAudioRawFrame chunks and TTSStoppedFrame on flushed."""
    captured: dict = {"init": None, "messages": [], "auth": None}
    audio_bytes = b"\x10\x20\x30\x40" * 512

    async def handler(ws):
        captured["auth"] = ws.request.headers.get("Authorization")
        async for raw in ws:
            msg = json.loads(raw)
            captured["messages"].append(msg)
            if msg.get("type") == "init":
                captured["init"] = msg
                await ws.send(json.dumps({"type": "ready", "session_id": "test-sess"}))
            elif msg.get("type") == "text":
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                            "sequence": 1,
                        }
                    )
                )
            elif msg.get("type") == "flush":
                await ws.send(json.dumps({"type": "flushed"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        base_url = f"ws://{host}:{port}"

        from pipecat.services.slng.tts import SlngTTSService

        tts = SlngTTSService(
            api_key="test-key",
            model="slng/deepgram/aura:2-en",
            voice="aura-2-thalia-en",
            base_url=base_url,
            sample_rate=24000,
        )

        down_frames, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello world"), SleepFrame(sleep=0.3)],
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSAudioRawFrame in frame_types
    assert TTSStoppedFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in audio_frames) == audio_bytes
    assert all(f.sample_rate == 24000 for f in audio_frames)

    assert captured["auth"] == "Bearer test-key"
    assert captured["init"]["type"] == "init"
    assert captured["init"]["config"]["sample_rate"] == 24000
    assert captured["init"]["config"]["encoding"] == "linear16"
    assert captured["init"]["voice"] == "aura-2-thalia-en"

    msg_types = [m["type"] for m in captured["messages"]]
    assert "init" in msg_types
    assert "text" in msg_types
    assert "flush" in msg_types


@pytest.mark.asyncio
async def test_slng_tts_ws_connect_failure_clears_websocket(monkeypatch):
    """Connect failure should leave _websocket as None."""
    from pipecat.services.slng.tts import SlngTTSService

    async def fake_connect(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("pipecat.services.slng.tts.websocket_connect", fake_connect)

    tts = SlngTTSService(api_key="test-key", sample_rate=24000)
    await tts._connect_websocket()

    assert tts._websocket is None
