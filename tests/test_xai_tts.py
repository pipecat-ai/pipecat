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
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
import websockets
from aiohttp import web
from websockets.asyncio.server import serve
from websockets.protocol import State

from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.xai.tts import (
    XAIHttpTTSService,
    XAITTSService,
    XAITTSSettings,
    XAIWebsocketTTSSettings,
    _xai_word_times,
)
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
    # A trailing space is appended so consecutive sentence segments don't glue.
    assert delta_msg["delta"] == "Hello from xAI. "


@pytest.mark.asyncio
async def test_xai_websocket_interruption_sends_text_clear():
    """On barge-in the WS service cancels via text.clear and keeps the socket open."""
    tts_service = XAITTSService(api_key="test-key", sample_rate=24000)

    websocket = AsyncMock()
    websocket.state = State.OPEN
    tts_service._websocket = websocket

    with (
        patch.object(tts_service, "_connect", new=AsyncMock()) as connect_spy,
        patch.object(tts_service, "_disconnect", new=AsyncMock()) as disconnect_spy,
    ):
        await tts_service.on_audio_context_interrupted("ctx-1")

    sent = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
    assert {"type": "text.clear"} in sent
    assert not connect_spy.called
    assert not disconnect_spy.called


@pytest.mark.asyncio
async def test_xai_websocket_settings_in_url():
    """Tunable settings appear as query params; booleans are lowercased for the URL."""
    captured: dict = {"request_path": None}

    async def handler(ws):
        captured["request_path"] = ws.request.path
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "text.done":
                    await ws.send(json.dumps({"type": "audio.done", "trace_id": "t"}))
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        base_url = f"ws://{host}:{port}/v1/tts"

        tts_service = XAITTSService(
            api_key="test-key",
            base_url=base_url,
            sample_rate=24000,
            settings=XAIWebsocketTTSSettings(
                speed=1.2,
                optimize_streaming_latency=2,
                text_normalization=True,
                with_timestamps=False,
            ),
        )

        await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello from xAI."), _SleepAfterSpeak(0.3)],
        )

    query = parse_qs(urlparse(captured["request_path"]).query)
    assert query["speed"] == ["1.2"]
    assert query["optimize_streaming_latency"] == ["2"]
    assert query["text_normalization"] == ["true"]
    assert query["with_timestamps"] == ["false"]


@pytest.mark.asyncio
async def test_xai_http_settings_in_body(aiohttp_client):
    """Tunable settings and language=auto appear in the HTTP request body."""
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
            settings=XAITTSSettings(
                language="auto",
                speed=1.2,
                optimize_streaming_latency=2,
                text_normalization=True,
            ),
        )

        await run_test(tts_service, frames_to_send=[TTSSpeakFrame(text="Hello from xAI.")])

    assert len(request_bodies) == 1
    body = request_bodies[0]
    assert body["language"] == "auto"
    assert body["speed"] == 1.2
    assert body["optimize_streaming_latency"] == 2
    assert body["text_normalization"] is True
    assert "with_timestamps" not in body


@pytest.mark.asyncio
async def test_xai_websocket_audio_clear_handled():
    """A server audio.clear ack is handled without producing an error."""
    audio_bytes = b"\x00\x01\x02\x03" * 1024

    async def handler(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "text.done":
                    await ws.send(json.dumps({"type": "audio.clear"}))
                    await ws.send(
                        json.dumps(
                            {
                                "type": "audio.delta",
                                "delta": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        )
                    )
                    await ws.send(json.dumps({"type": "audio.done", "trace_id": "t"}))
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        base_url = f"ws://{host}:{port}/v1/tts"

        tts_service = XAITTSService(api_key="test-key", base_url=base_url, sample_rate=24000)

        down_frames, up_frames = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello from xAI."), _SleepAfterSpeak(0.3)],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)


def test_xai_word_times_splits_and_carries_partials():
    """Character timings convert to absolute word starts, carrying partials across chunks."""
    # Chunk 1: "Hi there" — "there" has no terminating space, so it is partial.
    chars = ["H", "i", " ", "t", "h", "e", "r", "e"]
    times = [[i * 0.1, (i + 1) * 0.1] for i in range(len(chars))]
    word_times, partial, partial_start = _xai_word_times(chars, times)
    assert word_times == [("Hi", 0.0)]
    assert partial == "there"
    assert partial_start == pytest.approx(0.3)

    # Chunk 2: "! ok" — finishes "there!" then starts a new partial "ok". xAI times
    # are absolute across the utterance, so they are used as-is (no offset).
    chars2 = ["!", " ", "o", "k"]
    times2 = [[2.0, 2.05], [2.05, 2.1], [2.1, 2.15], [2.15, 2.2]]
    word_times2, partial2, partial2_start = _xai_word_times(
        chars2, times2, partial_word=partial, partial_word_start_time=partial_start
    )
    assert word_times2 == [("there!", pytest.approx(0.3))]
    assert partial2 == "ok"
    assert partial2_start == pytest.approx(2.1)


def test_xai_word_times_length_mismatch_is_safe():
    """A chars/times length mismatch yields no words and preserves the partial."""
    word_times, partial, partial_start = _xai_word_times(
        ["a", "b"], [[0.0, 0.1]], partial_word="x", partial_word_start_time=1.0
    )
    assert word_times == []
    assert partial == "x"
    assert partial_start == 1.0


@pytest.mark.asyncio
async def test_xai_websocket_emits_word_timestamps():
    """With with_timestamps enabled, the WS service emits TTSTextFrames per word."""
    audio_bytes = b"\x00\x01\x02\x03" * 1024
    chars = list("Hello world")
    times = [[i * 0.1, (i + 1) * 0.1] for i in range(len(chars))]

    async def handler(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "text.done":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "audio.delta",
                                "delta": base64.b64encode(audio_bytes).decode("ascii"),
                                "audio_timestamps": {"graph_chars": chars, "graph_times": times},
                                "audio_duration": 1.1,
                            }
                        )
                    )
                    await ws.send(json.dumps({"type": "audio.done", "trace_id": "t"}))
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        base_url = f"ws://{host}:{port}/v1/tts"

        tts_service = XAITTSService(
            api_key="test-key",
            base_url=base_url,
            sample_rate=24000,
            settings=XAIWebsocketTTSSettings(with_timestamps=True),
        )

        down_frames, _ = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello world"), _SleepAfterSpeak(0.3)],
        )

    words = [frame.text for frame in down_frames if isinstance(frame, TTSTextFrame)]
    joined = "".join(words)
    assert "Hello" in joined
    assert "world" in joined


@pytest.mark.asyncio
async def test_xai_websocket_word_timestamps_from_audioless_delta():
    """xAI sends timestamps in their own (audio-less) deltas; they must still emit words."""
    audio_bytes = b"\x00\x01\x02\x03" * 1024
    chars = list("Hi there")
    times = [[i * 0.1, (i + 1) * 0.1] for i in range(len(chars))]

    async def handler(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "text.done":
                    # Audio arrives first with no timestamps...
                    await ws.send(
                        json.dumps(
                            {
                                "type": "audio.delta",
                                "delta": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        )
                    )
                    # ...then timestamps arrive in a delta carrying no audio.
                    await ws.send(
                        json.dumps(
                            {
                                "type": "audio.delta",
                                "delta": "",
                                "audio_timestamps": {"graph_chars": chars, "graph_times": times},
                            }
                        )
                    )
                    await ws.send(json.dumps({"type": "audio.done", "trace_id": "t"}))
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        base_url = f"ws://{host}:{port}/v1/tts"

        tts_service = XAITTSService(api_key="test-key", base_url=base_url, sample_rate=24000)

        down_frames, _ = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hi there"), _SleepAfterSpeak(0.3)],
        )

    joined = "".join(frame.text for frame in down_frames if isinstance(frame, TTSTextFrame))
    assert "Hi" in joined
    assert "there" in joined


# Small helper imported lazily to avoid circular import in fixture-lite tests.
def _SleepAfterSpeak(duration: float):
    from pipecat.tests.utils import SleepFrame

    return SleepFrame(sleep=duration)


if __name__ == "__main__":
    unittest.main()
