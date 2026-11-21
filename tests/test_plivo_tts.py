#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for PlivoTTSService and PlivoHttpTTSService."""

import asyncio
import base64
import json

import aiohttp
import pytest
import websockets
from aiohttp import web
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.plivo.tts import (
    PlivoHttpTTSService,
    PlivoTTSService,
    calculate_word_times,
    language_to_plivo_language,
    parse_plivo_audio_frame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

AUDIO = b"\x00\x01\x02\x03" * 512


def _frame(context_id: str, audio: bytes) -> bytes:
    ctx = context_id.encode()
    return len(ctx).to_bytes(2, "little") + ctx + audio


def _speak_and_finish(text: str, sleep: float) -> list:
    # PlivoTTSService pauses frame processing while speaking and resumes on
    # BotStoppedSpeakingFrame, which an output transport would normally emit.
    return [TTSSpeakFrame(text=text), SleepFrame(sleep=sleep), BotStoppedSpeakingFrame()]


def test_parse_plivo_audio_frame():
    assert parse_plivo_audio_frame(_frame("c1", b"\x01\x02")) == ("c1", b"\x01\x02")
    with pytest.raises(ValueError):
        parse_plivo_audio_frame(b"\x05")
    with pytest.raises(ValueError):
        parse_plivo_audio_frame(b"\x05\x00ab")


def test_calculate_word_times():
    chars = list("Hi there.")
    start_ms = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    assert calculate_word_times(chars, start_ms) == [("Hi", 0.0), ("there.", 0.15)]
    assert calculate_word_times(["a"], []) == []


def test_language_mapping():
    assert language_to_plivo_language(Language.EN) == "en-US"
    assert language_to_plivo_language(Language.EN_IN) == "en-IN"
    assert language_to_plivo_language(Language.HI_IN) == "hi"


@pytest.mark.asyncio
async def test_plivo_websocket_tts_success():
    """The first text message opens the reply with settings; flush finishes it."""
    captured: dict = {"auth": None, "path": None, "messages": []}

    async def handler(ws):
        captured["auth"] = ws.request.headers.get("Authorization")
        captured["path"] = ws.request.path
        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                ctx = msg["context_id"]
                if msg.get("finish"):
                    await ws.send(_frame(ctx, AUDIO))
                    await ws.send(
                        json.dumps(
                            {
                                "type": "timestamps",
                                "context_id": ctx,
                                "chars": list("Hello Plivo."),
                                "start_ms": [i * 40 for i in range(12)],
                                "duration_ms": [40] * 12,
                            }
                        )
                    )
                    await ws.send(
                        json.dumps({"type": "done", "context_id": ctx, "reason": "complete"})
                    )
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = PlivoTTSService(
            auth_id="AUTHID",
            auth_token="token",
            url=f"ws://{host}:{port}/v1/Account/AUTHID/TTS/Realtime/",
            sample_rate=24000,
            settings=PlivoTTSService.Settings(voice="voice-1", pace=1.1),
        )
        down_frames, _ = await run_test(tts, frames_to_send=_speak_and_finish("Hello Plivo.", 0.5))

    types = [type(f) for f in down_frames]
    assert TTSStartedFrame in types
    assert TTSStoppedFrame in types
    audio = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in audio) == AUDIO
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio)
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words == ["Hello", "Plivo."]

    expected_auth = "Basic " + base64.b64encode(b"AUTHID:token").decode()
    assert captured["auth"] == expected_auth
    assert captured["path"] == "/v1/Account/AUTHID/TTS/Realtime/"

    first, *rest = captured["messages"]
    assert first["type"] == "text"
    assert first["text"] == "Hello Plivo."
    assert first["voice_id"] == "voice-1"
    assert first["language"] == "en-IN"
    assert first["pace"] == 1.1
    assert first["timestamps"] is True
    assert first["output_format"] == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 24000,
    }
    assert "volume" not in first and "lexicon_id" not in first
    # Later messages for the same reply must only carry type/context_id/text/finish.
    assert rest, "expected a finishing message"
    for msg in rest:
        assert set(msg) <= {"type", "context_id", "text", "finish"}
        assert msg["context_id"] == first["context_id"]
    assert rest[-1]["finish"] is True


@pytest.mark.asyncio
async def test_plivo_websocket_tts_error_event():
    """A Plivo error event surfaces as an ErrorFrame and ends the context."""

    async def handler(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                await ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "code": "invalid_request",
                            "message": "bad voice",
                            "context_id": msg["context_id"],
                        }
                    )
                )
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = PlivoTTSService(
            auth_id="AUTHID",
            auth_token="token",
            url=f"ws://{host}:{port}/",
            sample_rate=24000,
            settings=PlivoTTSService.Settings(voice="bad"),
        )
        _, up_frames = await run_test(tts, frames_to_send=_speak_and_finish("Hello.", 0.3))

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "bad voice" in errors[0].error


@pytest.mark.asyncio
async def test_plivo_websocket_tts_cancel_on_interruption():
    """Interrupting the active context sends a cancel for it."""
    received: list[dict] = []
    cancel_seen = asyncio.Event()

    async def handler(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                received.append(msg)
                if msg["type"] == "cancel":
                    cancel_seen.set()
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = PlivoTTSService(
            auth_id="AUTHID",
            auth_token="token",
            url=f"ws://{host}:{port}/",
            sample_rate=24000,
            settings=PlivoTTSService.Settings(voice="voice-1"),
        )
        # Exercise the hook directly: open a context, then interrupt it.
        tts._websocket = await connect(f"ws://{host}:{port}/")
        tts._open_context_ids.add("ctx-1")
        await tts.on_audio_context_interrupted("ctx-1")
        await asyncio.wait_for(cancel_seen.wait(), timeout=1.0)
        await tts._websocket.close()

    assert {"type": "cancel", "context_id": "ctx-1"} in received


@pytest.mark.asyncio
async def test_plivo_http_tts_success(aiohttp_client):
    """The Batch API request body is correct and odd-sized chunks stay sample-aligned."""
    bodies = []
    auth_headers = []

    async def handler(request):
        bodies.append(await request.json())
        auth_headers.append(request.headers.get("Authorization"))
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "application/octet-stream"}
        )
        await response.prepare(request)
        await response.write(AUDIO[:1001])
        await asyncio.sleep(0.01)
        await response.write(AUDIO[1001:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/Account/AUTHID/TTS/", handler)
    client = await aiohttp_client(app)

    async with aiohttp.ClientSession() as session:
        tts = PlivoHttpTTSService(
            auth_id="AUTHID",
            auth_token="token",
            url=str(client.make_url("/v1/Account/AUTHID/TTS/")),
            sample_rate=16000,
            aiohttp_session=session,
            settings=PlivoHttpTTSService.Settings(voice="voice-1", language=Language.HI),
        )
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Namaste.")])

    types = [type(f) for f in down_frames]
    assert TTSStartedFrame in types and TTSStoppedFrame in types
    audio = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert all(len(f.audio) % 2 == 0 for f in audio)
    assert b"".join(f.audio for f in audio) == AUDIO

    assert auth_headers[0] == "Basic " + base64.b64encode(b"AUTHID:token").decode()
    assert bodies[0] == {
        "text": "Namaste.",
        "voice_id": "voice-1",
        "language": "hi",
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
    }


@pytest.mark.asyncio
async def test_plivo_http_tts_error(aiohttp_client):
    """A non-200 response produces an ErrorFrame."""

    async def handler(request):
        return web.json_response({"error": "language_not_supported"}, status=400)

    app = web.Application()
    app.router.add_post("/tts", handler)
    client = await aiohttp_client(app)

    async with aiohttp.ClientSession() as session:
        tts = PlivoHttpTTSService(
            auth_id="AUTHID",
            auth_token="token",
            url=str(client.make_url("/tts")),
            sample_rate=16000,
            aiohttp_session=session,
            settings=PlivoHttpTTSService.Settings(voice="voice-1"),
        )
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors and "language_not_supported" in errors[0].error
