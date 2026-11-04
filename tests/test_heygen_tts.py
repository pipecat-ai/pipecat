#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for HeyGenTTSService and HeyGenHttpTTSService."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
import websockets
from aiohttp import web
from websockets.asyncio.server import serve
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.heygen.tts import (
    HEYGEN_DEFAULT_VOICE_ID,
    HeyGenHttpTTSService,
    HeyGenTTSService,
    language_to_heygen_language,
)
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

STREAM_PATH = "/v3/models/audio/tts/stream"


def _make_app(requests: list, audio: bytes = b"", status: int = 200, error: str = ""):
    async def handler(request):
        requests.append({"headers": dict(request.headers), "body": await request.json()})
        if status != 200:
            return web.json_response({"error": {"message": error}}, status=status)

        response = web.StreamResponse(
            status=200, headers={"Content-Type": "audio/pcm", "X-Request-Id": "req-123"}
        )
        await response.prepare(request)
        # Split mid-sample to check that emitted frames stay sample-aligned.
        await response.write(audio[:2047])
        await asyncio.sleep(0.01)
        await response.write(audio[2047:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post(STREAM_PATH, handler)
    return app


@pytest.mark.asyncio
async def test_heygen_tts_streams_pcm(aiohttp_client):
    """The service sends the documented request and emits sample-aligned PCM frames."""
    requests = []
    pcm_audio = b"\x00\x01\x02\x03" * 1024
    client = await aiohttp_client(_make_app(requests, pcm_audio))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=24000,
        )
        down_frames, _ = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hello from HeyGen.")]
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio_frames)
    assert all(len(f.audio) % 2 == 0 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == pcm_audio

    assert len(requests) == 1
    assert requests[0]["headers"]["x-api-key"] == "test-key"
    assert requests[0]["body"] == {
        "text": "Hello from HeyGen.",
        "voice_id": HEYGEN_DEFAULT_VOICE_ID,
        "model": "heygen-voice-1-turbo",
        "output_format": {"encoding": "pcm", "sample_rate_hertz": 24000},
    }


@pytest.mark.asyncio
async def test_heygen_tts_settings_in_body(aiohttp_client):
    """Voice, model and language settings reach the request body."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 2048))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=16000,
            settings=HeyGenHttpTTSService.Settings(
                voice="my-voice", model="heygen-voice-1", language=Language.EN_US
            ),
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    body = requests[0]["body"]
    assert body["voice_id"] == "my-voice"
    assert body["model"] == "heygen-voice-1"
    assert body["language"] == "en"
    assert body["output_format"]["sample_rate_hertz"] == 16000


@pytest.mark.asyncio
async def test_heygen_tts_unsupported_sample_rate_is_resampled(aiohttp_client):
    """A pipeline rate HeyGen doesn't offer requests 24 kHz and resamples to the pipeline rate."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 24000))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=32000,
        )
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert requests[0]["body"]["output_format"]["sample_rate_hertz"] == 24000
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 32000 for f in audio_frames)


@pytest.mark.asyncio
async def test_heygen_tts_http_error(aiohttp_client):
    """A non-200 response produces an ErrorFrame with the status and message."""
    requests = []
    client = await aiohttp_client(_make_app(requests, status=429, error="Request limit reached."))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=24000,
        )
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "429" in errors[0].error
    assert "Request limit reached." in errors[0].error


def test_language_mapping():
    """Regional variants resolve to HeyGen's base ISO 639-1 codes."""
    assert language_to_heygen_language(Language.EN) == "en"
    assert language_to_heygen_language(Language.EN_GB) == "en"
    assert language_to_heygen_language(Language.ZH_HK) == "zh"
    assert language_to_heygen_language(Language.PT_BR) == "pt"


# --- HeyGenTTSService (WebSocket) ---


def _alignment(text: str) -> dict:
    chars = list(text)
    return {
        "chars": chars,
        "charStartTimesMs": [i * 50 for i in range(len(chars))],
        "charDurationsMs": [50 for _ in chars],
    }


async def _serve_heygen_ws(captured: dict, reply_with_error: bool = False):
    """A fake multi-stream-input socket: audio for each text, isFinal on close_context."""
    audio = b"\x00\x01" * 2400

    async def handler(ws):
        captured["path"] = ws.request.path
        captured["api_key"] = ws.request.headers.get("x-api-key")
        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                ctx = msg.get("context_id")
                if reply_with_error and msg.get("text", "").strip():
                    await ws.send(
                        json.dumps(
                            {
                                "message": "Insufficient credits.",
                                "error": {"code": "usage_limit_reached"},
                            }
                        )
                    )
                    await ws.close(1008, "usage_limit_reached")
                    return
                if msg.get("text", "").strip():
                    await ws.send(
                        json.dumps(
                            {
                                "audio": base64.b64encode(audio).decode(),
                                "alignment": _alignment(msg["text"]),
                                "contextId": ctx,
                                "isFinal": None,
                            }
                        )
                    )
                if msg.get("close_context"):
                    await ws.send(json.dumps({"isFinal": True, "contextId": ctx}))
                if msg.get("close_socket"):
                    await ws.close()
                    return
        except websockets.ConnectionClosed:
            pass

    return serve(handler, "127.0.0.1", 0), audio


@pytest.mark.asyncio
async def test_heygen_websocket_streams_audio_and_words():
    """The service connects with the v3 parameters and turns alignment into word frames."""
    captured: dict = {"messages": []}
    server_cm, audio = await _serve_heygen_ws(captured)

    async with server_cm as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = HeyGenTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}",
            sample_rate=24000,
            settings=HeyGenTTSService.Settings(language=Language.EN_US),
        )
        down_frames, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from HeyGen."), SleepFrame(sleep=0.5)],
        )

    assert captured["api_key"] == "test-key"
    parsed = urlparse(captured["path"])
    assert parsed.path == "/v3/models/audio/tts/multi-stream-input"
    assert parse_qs(parsed.query) == {
        "voice_id": [HEYGEN_DEFAULT_VOICE_ID],
        "model": ["heygen-voice-1-turbo"],
        "output_format": ["pcm_24000"],
        "auto_mode": ["true"],
        "language": ["en"],
    }

    sent = captured["messages"]
    context_id = sent[0]["context_id"]
    assert sent[0] == {"text": " ", "context_id": context_id}
    assert {"text": "Hello from HeyGen. ", "context_id": context_id} in sent
    assert {"context_id": context_id, "close_context": True} in sent

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in audio_frames) == audio
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words == ["Hello", "from", "HeyGen."]


@pytest.mark.asyncio
async def test_heygen_websocket_error_message_becomes_error_frame():
    """HeyGen's error message before a close surfaces as an ErrorFrame with its text."""
    captured: dict = {"messages": []}
    server_cm, _ = await _serve_heygen_ws(captured, reply_with_error=True)

    async with server_cm as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = HeyGenTTSService(api_key="test-key", url=f"ws://{host}:{port}", sample_rate=24000)
        _, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello."), SleepFrame(sleep=0.5)],
        )

    errors = [f.error for f in up_frames if isinstance(f, ErrorFrame)]
    assert any("Insufficient credits." in e for e in errors)


@pytest.mark.asyncio
async def test_heygen_websocket_interruption_closes_context():
    """On barge-in the service closes the HeyGen context and keeps the socket open."""
    tts = HeyGenTTSService(api_key="test-key", sample_rate=24000)
    websocket = AsyncMock()
    websocket.state = State.OPEN
    tts._websocket = websocket

    await tts.on_audio_context_interrupted("ctx-1")

    sent = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
    assert {"context_id": "ctx-1", "close_context": True} in sent


def test_heygen_websocket_token_mode_disables_auto_mode():
    """Token streaming leaves generation to HeyGen's chunk schedule."""
    tts = HeyGenTTSService(api_key="k", text_aggregation_mode=TextAggregationMode.TOKEN)
    assert tts._auto_mode is False
    assert HeyGenTTSService(api_key="k")._auto_mode is True
