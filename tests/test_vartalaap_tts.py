#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for VartalaapTTSService and VartalaapHttpTTSService against a fake Vartalaap server.

The fake speaks the real wire protocol: one JSON message per utterance in, then ``start``,
binary PCM frames, ``first_audio_ms`` and ``end`` events out, all tagged with the utterance's
``request_id``; errors are ``error`` events with a ``status``.
"""

import asyncio
import json
from dataclasses import dataclass, field

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.services.vartalaap.tts import (
    VARTALAAP_SAMPLE_RATE,
    VartalaapHttpTTSService,
    VartalaapTTSService,
    _utterance_body,
)
from pipecat.tests.utils import SleepFrame, run_test

FRAME = b"\x01\x00" * 1920  # one 80 ms pcm16 frame at 24 kHz


@dataclass
class _FakeVartalaap:
    """A Vartalaap WebSocket server: utterances are served in order, each as start/audio/end."""

    frames_per_utterance: int = 2
    # request_ids to answer with this error event instead of audio (by arrival index)
    error_at: dict[int, dict] = field(default_factory=dict)
    received: list[dict] = field(default_factory=list)
    headers: list[dict] = field(default_factory=list)
    connections: int = 0
    # send an error of an utterance that is not the client's as soon as it connects
    stale_error: bool = False

    async def handle(self, websocket):
        self.connections += 1
        self.headers.append(dict(websocket.request.headers))
        if self.stale_error:
            await websocket.send(
                json.dumps({"type": "error", "request_id": "old", "status": 503, "error": "busy"})
            )
        try:
            async for raw in websocket:
                msg = json.loads(raw)
                self.received.append(msg)
                index = len(self.received) - 1
                rid = msg["request_id"]
                if index in self.error_at:
                    await websocket.send(
                        json.dumps({"type": "error", "request_id": rid, **self.error_at[index]})
                    )
                    continue
                await websocket.send(
                    json.dumps(
                        {
                            "type": "start",
                            "request_id": rid,
                            "voice": msg["voice"],
                            "format": "pcm16",
                            "chunks": 1,
                            "normalized": msg["text"],
                        }
                    )
                )
                await websocket.send(json.dumps({"type": "chunk", "request_id": rid, "chunk": 0}))
                for i in range(self.frames_per_utterance):
                    await websocket.send(FRAME)
                    if i == 0:
                        await websocket.send(
                            json.dumps({"type": "first_audio_ms", "request_id": rid, "ms": 21.5})
                        )
                await websocket.send(
                    json.dumps({"type": "end", "request_id": rid, "audio_s": 0.16, "ttfa_ms": 18})
                )
        except ConnectionClosed:
            pass


async def _serve(fake: _FakeVartalaap):
    server = await serve(fake.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}/v1/tts/stream"


def _ws_service(url: str, **settings) -> VartalaapTTSService:
    return VartalaapTTSService(
        api_key="vt_live_test", url=url, settings=VartalaapTTSService.Settings(**settings)
    )


@pytest.mark.asyncio
async def test_turn_is_started_spoken_and_stopped_in_order():
    fake = _FakeVartalaap()
    server, url = await _serve(fake)
    try:
        tts = _ws_service(url, voice="sahil", digits="hi")
        down, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame("नमस्ते, आपका loan approve हो गया है।"), SleepFrame(0.3)],
        )
    finally:
        server.close()
        await server.wait_closed()

    types = [type(f) for f in down]
    assert TTSStartedFrame in types and TTSStoppedFrame in types and TTSTextFrame in types
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert len(audio) == 2
    assert all(f.sample_rate == VARTALAAP_SAMPLE_RATE and f.audio == FRAME for f in audio)
    started = types.index(TTSStartedFrame)
    stopped = types.index(TTSStoppedFrame)
    assert started < types.index(TTSAudioRawFrame) < stopped

    # the message carries the settings, the key rides on the handshake
    assert len(fake.received) == 1
    msg = fake.received[0]
    assert msg["voice"] == "sahil" and msg["digits"] == "hi" and msg["format"] == "pcm16"
    assert "style" not in msg  # neutral expression sends nothing
    assert fake.headers[0]["authorization"] == "Bearer vt_live_test"


@pytest.mark.asyncio
async def test_each_sentence_is_its_own_utterance_and_the_turn_closes_after_the_last():
    fake = _FakeVartalaap()
    server, url = await _serve(fake)
    try:
        tts = _ws_service(url)
        down, _ = await run_test(
            tts,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("पहला वाक्य "),
                LLMTextFrame("है। दूसरा "),
                LLMTextFrame("वाक्य है। तीसरा वाक्य है।"),
                LLMFullResponseEndFrame(),
                SleepFrame(0.5),
            ],
        )
    finally:
        server.close()
        await server.wait_closed()

    assert [m["text"].strip() for m in fake.received] == [
        "पहला वाक्य है।",
        "दूसरा वाक्य है।",
        "तीसरा वाक्य है।",
    ]
    assert len({m["request_id"] for m in fake.received}) == 3
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert len(audio) == 6
    stops = [f for f in down if isinstance(f, TTSStoppedFrame)]
    assert len(stops) == 1
    assert down.index(stops[0]) > down.index(audio[-1])


@pytest.mark.asyncio
async def test_settings_update_applies_to_the_next_utterance_without_reconnecting():
    fake = _FakeVartalaap()
    server, url = await _serve(fake)
    try:
        tts = _ws_service(url)
        await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame("Hello there."),
                SleepFrame(0.2),
                TTSUpdateSettingsFrame(
                    delta=VartalaapTTSService.Settings(
                        voice="sahil", expression="excited", speed=1.2, profile="telephony"
                    )
                ),
                TTSSpeakFrame("Hello again."),
                SleepFrame(0.3),
            ],
        )
    finally:
        server.close()
        await server.wait_closed()

    first, second = fake.received
    assert first["voice"] == "saanvi" and "style" not in first and "speed" not in first
    assert second["voice"] == "sahil"
    assert second["style"] == "excited"  # the API's name for the expression
    assert second["speed"] == 1.2
    assert second["profile"] == "telephony"
    assert fake.connections == 1


@pytest.mark.asyncio
async def test_a_server_error_is_reported_and_the_turn_still_closes():
    fake = _FakeVartalaap(error_at={0: {"status": 503, "retry_after": 1, "error": "busy"}})
    server, url = await _serve(fake)
    try:
        tts = _ws_service(url)
        down, up = await run_test(
            tts, frames_to_send=[TTSSpeakFrame("Hello there."), SleepFrame(0.3)]
        )
    finally:
        server.close()
        await server.wait_closed()

    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert errors and "503" in errors[0].error and "busy" in errors[0].error
    assert not [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert [type(f) for f in down].count(TTSStoppedFrame) == 1


@pytest.mark.asyncio
async def test_a_stale_error_of_another_utterance_is_ignored():
    fake = _FakeVartalaap(stale_error=True)
    server, url = await _serve(fake)
    try:
        tts = _ws_service(url)
        down, up = await run_test(
            tts, frames_to_send=[TTSSpeakFrame("Hello there."), SleepFrame(0.3)]
        )
    finally:
        server.close()
        await server.wait_closed()

    assert not [f for f in up if isinstance(f, ErrorFrame)]
    assert len([f for f in down if isinstance(f, TTSAudioRawFrame)]) == 2


def test_utterance_body_sends_only_what_differs_from_the_server_defaults():
    settings = VartalaapTTSService(api_key="k")._settings
    assert _utterance_body(settings, text="hi", request_id="r") == {
        "text": "hi",
        "voice": "saanvi",
        "format": "pcm16",
        "request_id": "r",
        "digits": "auto",
    }
    settings = VartalaapTTSService(
        api_key="k",
        settings=VartalaapTTSService.Settings(
            voice="meera",
            expression="whisper",
            digits="en",
            speed=0.9,
            temperature=0.4,
            seed=3,
            postfx=False,
            profile="telephony",
            raw=True,
        ),
    )._settings
    body = _utterance_body(settings, text="hi", request_id="r")
    assert body["voice"] == "meera"
    assert body["style"] == "whisper"
    assert body["digits"] == "en"
    assert body["speed"] == 0.9
    assert body["temperature"] == 0.4
    assert body["seed"] == 3
    assert body["postfx"] is False
    assert body["profile"] == "telephony"
    assert body["raw"] is True
    assert "model" not in body  # the server picks the model from the voice


def test_settings_aliases_map_the_api_names():
    delta = VartalaapTTSService.Settings.from_mapping({"style": "sad", "pace": 1.3})
    assert delta.expression == "sad"
    assert delta.speed == 1.3


class _ChunkedBody:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_any(self):
        for c in self._chunks:
            await asyncio.sleep(0)
            yield c


class _Response:
    def __init__(self, status, chunks=(), text=""):
        self.status = status
        self.content = _ChunkedBody(chunks)
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, *, json, headers):
        self.calls.append((url, json, headers))
        return self.response


@pytest.mark.asyncio
async def test_http_service_streams_the_chunked_body_as_it_arrives():
    session = _Session(_Response(200, chunks=[FRAME[:100], FRAME[100:], FRAME]))
    tts = VartalaapHttpTTSService(
        api_key="vt_live_test",
        aiohttp_session=session,
        sample_rate=VARTALAAP_SAMPLE_RATE,
        settings=VartalaapHttpTTSService.Settings(voice="sahil", expression="excited"),
    )
    down, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame("Hello there."), SleepFrame(0.2)])

    url, payload, headers = session.calls[0]
    assert url == "https://api.vartaloop.ai/v1/tts"
    assert payload["voice"] == "sahil" and payload["style"] == "excited"
    assert payload["format"] == "pcm16" and payload["stream"] is True
    assert headers["Authorization"] == "Bearer vt_live_test"
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in audio) == FRAME + FRAME
    # the first chunk is forwarded without waiting for the rest
    assert len(audio) == 3
    types = [type(f) for f in down]
    assert (
        types.index(TTSStartedFrame) < types.index(TTSAudioRawFrame) < types.index(TTSStoppedFrame)
    )


@pytest.mark.asyncio
async def test_http_service_reports_a_non_200():
    session = _Session(_Response(401, text='{"error":"invalid api key"}'))
    tts = VartalaapHttpTTSService(api_key="bad", aiohttp_session=session)
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("Hello there."), SleepFrame(0.2)])
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert errors and "401" in errors[0].error
    assert not [f for f in down if isinstance(f, TTSAudioRawFrame)]
