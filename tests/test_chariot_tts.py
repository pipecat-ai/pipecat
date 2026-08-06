#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Chariot TTS service."""

import asyncio
import json

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close
from websockets.protocol import State

from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.chariot.tts import (
    CHARIOT_DEFAULT_VOICE,
    CHARIOT_SAMPLE_RATE,
    ChariotTTSService,
    ChariotTTSSettings,
)


def _service(**kwargs) -> ChariotTTSService:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("voice_id", "test-voice")
    return ChariotTTSService(**kwargs)


class _FakeWebsocket:
    """Collects sent messages; optionally yields scripted incoming messages."""

    def __init__(self, incoming=None, raise_after=None):
        self.state = State.OPEN
        self.sent: list[str] = []
        self._incoming = list(incoming or [])
        self._raise_after = raise_after

    async def send(self, message: str):
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._incoming:
            return self._incoming.pop(0)
        if self._raise_after is not None:
            raise self._raise_after
        raise StopAsyncIteration


def test_default_voice_when_none_given():
    service = ChariotTTSService(api_key="test-key")
    assert service._settings.voice == CHARIOT_DEFAULT_VOICE


def test_settings_voice_wins_over_voice_id():
    service = _service(
        voice_id="direct-voice",
        settings=ChariotTTSSettings(voice="settings-voice"),
    )
    assert service._settings.voice == "settings-voice"


def test_websocket_url():
    service = _service(optimize_streaming_latency=2)
    url = service._websocket_url()
    assert url.startswith("wss://api.chariot.in/v1/tts/ws?")
    assert "voice_id=test-voice" in url
    assert "response_format=pcm" in url
    assert "idle_timeout=300" in url
    assert "optimize_streaming_latency=2" in url


def test_base_url_override():
    service = _service(base_url="wss://example.test/")
    assert service._websocket_url().startswith("wss://example.test/v1/tts/ws?")


def test_default_sample_rate_is_chariot_rate():
    service = _service()
    assert service._output_sample_rate == CHARIOT_SAMPLE_RATE


@pytest.mark.asyncio
async def test_send_text_appends_then_flushes():
    service = _service()
    ws = _FakeWebsocket()
    service._websocket = ws

    await service._send_text("Namaste!")

    assert [json.loads(m)["type"] for m in ws.sent] == ["input.text", "input.flush"]
    assert json.loads(ws.sent[0])["text"] == "Namaste!"


@pytest.mark.asyncio
async def test_receive_routes_audio_and_stop_to_context():
    service = _service()
    appended = []
    removed = []

    service.get_active_audio_context_id = lambda: "ctx-1"
    service.audio_context_available = lambda context_id: True

    async def _append(context_id, frame):
        appended.append((context_id, frame))

    async def _remove(context_id):
        removed.append(context_id)

    async def _noop():
        pass

    service.append_to_audio_context = _append
    service.remove_audio_context = _remove
    service.stop_ttfb_metrics = _noop

    ws = _FakeWebsocket(
        incoming=[
            json.dumps({"type": "audio.start", "sample_rate": 22050}),
            b"\x00\x01" * 32,
            b"\x02\x03" * 32,
            json.dumps({"type": "audio.done"}),
        ]
    )
    service._get_websocket = lambda: ws

    await service._receive_messages()

    audio_frames = [f for _, f in appended if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) == 2
    # The server-declared rate from audio.start is applied to every frame.
    assert all(f.sample_rate == 22050 for f in audio_frames)
    assert all(context_id == "ctx-1" for context_id, _ in appended)

    stop_frames = [f for _, f in appended if isinstance(f, TTSStoppedFrame)]
    assert len(stop_frames) == 1
    assert removed == ["ctx-1"]


@pytest.mark.asyncio
async def test_receive_reports_exhausted_credits():
    service = _service()
    errors = []

    async def _push_error(error_msg=None, **kwargs):
        errors.append(error_msg)

    service.push_error = _push_error
    service.get_active_audio_context_id = lambda: None

    closed = ConnectionClosedError(Close(4402, "no credits"), None)
    ws = _FakeWebsocket(raise_after=closed)
    service._get_websocket = lambda: ws

    await service._receive_messages()

    assert len(errors) == 1
    assert "credit balance exhausted" in errors[0]


@pytest.mark.asyncio
async def test_receive_reraises_other_close_errors():
    service = _service()
    service.get_active_audio_context_id = lambda: None

    closed = ConnectionClosedError(Close(1011, "server error"), None)
    ws = _FakeWebsocket(raise_after=closed)
    service._get_websocket = lambda: ws

    with pytest.raises(ConnectionClosedError):
        await service._receive_messages()


@pytest.mark.asyncio
async def test_flush_audio_sends_input_flush():
    service = _service()
    ws = _FakeWebsocket()
    service._websocket = ws

    await service.flush_audio()

    assert [json.loads(m)["type"] for m in ws.sent] == ["input.flush"]


def test_split_text_reconstructs_exactly():
    from pipecat.services.chariot.tts import _split_text

    text = "All work and no play makes Jack a dull boy. " * 50
    pieces = _split_text(text, 500)
    assert "".join(pieces) == text
    assert all(len(p) <= 500 for p in pieces)
    # A single over-long token is hard-split rather than looping forever.
    assert _split_text("x" * 1200, 500) == ["x" * 500, "x" * 500, "x" * 200]


@pytest.mark.asyncio
async def test_long_text_is_chunked_before_flush():
    service = _service()
    ws = _FakeWebsocket()
    service._websocket = ws

    long_text = "All work and no play makes Jack a dull boy. " * 50  # 2200 chars
    await service._send_text(long_text)

    messages = [json.loads(m) for m in ws.sent]
    # Strict alternation: every buffered segment is flushed before the next.
    assert [m["type"] for m in messages] == ["input.text", "input.flush"] * (len(messages) // 2)
    texts = [m["text"] for m in messages if m["type"] == "input.text"]
    assert len(texts) > 1  # actually segmented
    assert all(len(t) <= 500 for t in texts)
    assert "".join(texts) == long_text
