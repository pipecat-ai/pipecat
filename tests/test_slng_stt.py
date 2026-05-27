#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Tests for SlngSTTService."""

import asyncio
import base64
import json

import pytest
from websockets.asyncio.server import serve
from websockets.protocol import State

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test


@pytest.mark.asyncio
async def test_slng_stt_ws_partial_and_final():
    """WS STT should emit InterimTranscriptionFrame then TranscriptionFrame."""
    captured: dict = {"init": None, "audio_msgs": [], "auth": None, "finalize_sent": False}

    async def handler(ws):
        captured["auth"] = ws.request.headers.get("Authorization")
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "init":
                captured["init"] = msg
                await ws.send(json.dumps({"type": "ready", "session_id": "test-sess"}))
            elif msg.get("type") == "audio":
                captured["audio_msgs"].append(msg)
                await ws.send(
                    json.dumps({"type": "partial_transcript", "transcript": "hello", "confidence": 0.8})
                )
                await asyncio.sleep(0.05)
                await ws.send(
                    json.dumps(
                        {
                            "type": "final_transcript",
                            "transcript": "hello world",
                            "confidence": 0.95,
                            "language": "en",
                        }
                    )
                )
            elif msg.get("type") == "finalize":
                captured["finalize_sent"] = True

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        base_url = f"ws://{host}:{port}"

        from pipecat.services.slng.stt import SlngSTTService

        stt = SlngSTTService(
            api_key="test-key",
            model="slng/deepgram/nova:3-en",
            base_url=base_url,
            sample_rate=16000,
        )

        from pipecat.frames.frames import AudioRawFrame

        audio = AudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)

        down_frames, _ = await run_test(
            stt,
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                audio,
                SleepFrame(sleep=0.2),
                VADUserStoppedSpeakingFrame(),
            ],
        )

    frame_types = [type(f) for f in down_frames]
    assert InterimTranscriptionFrame in frame_types
    assert TranscriptionFrame in frame_types

    interim = next(f for f in down_frames if isinstance(f, InterimTranscriptionFrame))
    assert interim.text == "hello"

    final = next(f for f in down_frames if isinstance(f, TranscriptionFrame))
    assert final.text == "hello world"

    assert captured["auth"] == "Bearer test-key"
    assert captured["init"] is not None
    assert captured["init"]["type"] == "init"
    assert captured["init"]["config"]["sample_rate"] == 16000
    assert captured["init"]["config"]["encoding"] == "linear16"
    assert captured["init"]["config"]["enable_partials"] is True

    assert captured["audio_msgs"], "audio messages should have been sent"
    audio_msg = captured["audio_msgs"][0]
    assert audio_msg["type"] == "audio"
    assert base64.b64decode(audio_msg["data"]) == b"\x00" * 320


@pytest.mark.asyncio
async def test_slng_stt_ws_error_message_pushes_no_transcription():
    """WS STT should not emit transcription frames when server sends error."""

    async def handler(ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "init":
                await ws.send(json.dumps({"type": "ready", "session_id": "err-sess"}))
            elif msg.get("type") == "audio":
                await ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "code": "provider_error",
                            "message": "upstream unavailable",
                        }
                    )
                )

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        base_url = f"ws://{host}:{port}"

        from pipecat.frames.frames import AudioRawFrame
        from pipecat.services.slng.stt import SlngSTTService

        stt = SlngSTTService(
            api_key="test-key",
            model="slng/deepgram/nova:3-en",
            base_url=base_url,
            sample_rate=16000,
        )

        audio = AudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)

        down_frames, _ = await run_test(
            stt,
            frames_to_send=[VADUserStartedSpeakingFrame(), audio, SleepFrame(sleep=0.2)],
        )

    frame_types = [type(f) for f in down_frames]
    assert InterimTranscriptionFrame not in frame_types
    assert TranscriptionFrame not in frame_types


@pytest.mark.asyncio
async def test_slng_stt_ws_connect_failure_clears_websocket(monkeypatch):
    """Connect failure should leave _websocket as None."""
    from pipecat.services.slng.stt import SlngSTTService

    async def fake_connect(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("pipecat.services.slng.stt.websocket_connect", fake_connect)

    stt = SlngSTTService(api_key="test-key", model="slng/deepgram/nova:3-en", sample_rate=16000)
    await stt._connect_websocket()

    assert stt._websocket is None
