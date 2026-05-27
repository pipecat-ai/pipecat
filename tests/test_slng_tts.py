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


@pytest.mark.asyncio
async def test_slng_http_tts_streams_audio_chunks():
    """HTTP TTS should stream audio chunks and emit TTSAudioRawFrame frames."""
    from unittest.mock import AsyncMock, MagicMock

    from pipecat.services.slng.tts import SlngHttpTTSService

    audio_bytes = b"\xAA\xBB" * 1024

    async def fake_iter_bytes(chunk_size=None):
        half = len(audio_bytes) // 2
        yield audio_bytes[:half]
        yield audio_bytes[half:]

    mock_response = MagicMock()
    mock_response.iter_bytes = fake_iter_bytes

    mock_streaming_ctx = MagicMock()
    mock_streaming_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_streaming_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_streaming_resource = MagicMock()
    mock_streaming_resource.create = MagicMock(return_value=mock_streaming_ctx)

    mock_tts_resource = MagicMock()
    mock_tts_resource.with_streaming_response = mock_streaming_resource

    mock_client = MagicMock()
    mock_client.text_to_speech = mock_tts_resource

    tts = SlngHttpTTSService(
        api_key="test-key",
        model="slng/deepgram/aura:2-en",
        voice="aura-2-thalia-en",
        sample_rate=24000,
    )
    tts._client = mock_client

    down_frames, _ = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="Hello from HTTP TTS.")],
    )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSAudioRawFrame in frame_types
    assert TTSStoppedFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in audio_frames) == audio_bytes
    assert all(f.sample_rate == 24000 for f in audio_frames)

    mock_streaming_resource.create.assert_called_once_with(
        "slng/deepgram/aura:2-en",
        text="Hello from HTTP TTS.",
        voice="aura-2-thalia-en",
    )
