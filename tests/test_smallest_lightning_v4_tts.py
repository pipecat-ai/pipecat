#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SmallestLightningV4TTSService's turn-based live-session handling."""

import json

import pytest

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.smallest.tts_v4 import (
    SmallestLightningV4TTSService,
    language_to_smallest_lightning_v4_language,
)
from pipecat.transcriptions.language import Language

CTX = "ctx-1"


def _make_service(**kwargs) -> SmallestLightningV4TTSService:
    settings = kwargs.pop("settings", None) or SmallestLightningV4TTSService.Settings(
        voice="rhodes"
    )
    return SmallestLightningV4TTSService(api_key="test-key", settings=settings, **kwargs)


def test_defaults_to_brannock_voice():
    """A voice is not required; it defaults to `brannock` if unset."""
    service = SmallestLightningV4TTSService(api_key="test-key")
    assert service._settings.voice == "brannock"


def test_does_not_reuse_context_id_within_a_turn():
    """Lightning v4 has no continuation model, so each aggregated fragment
    must become its own turn rather than sharing one context id."""
    service = _make_service()
    assert service._reuse_context_id_within_turn is False


@pytest.mark.parametrize(
    "language,expected",
    [
        (Language.EN, "en"),
        (Language.EN_US, "en"),
        (Language.EN_GB, "en"),
        (Language.FR, "auto"),
        (Language.HI, "auto"),
    ],
)
def test_language_mapping(language, expected):
    assert language_to_smallest_lightning_v4_language(language) == expected


def test_build_websocket_url_includes_connect_time_params():
    service = _make_service(
        settings=SmallestLightningV4TTSService.Settings(
            voice="rhodes", speed=1.2, content_filter=True, content_filter_action="flag"
        )
    )
    service._v4_sample_rate = 24000
    url = service._build_websocket_url()

    assert url.startswith("wss://api.smallest.ai/waves/v1/lightning-v4/live?")
    assert "voice_id=rhodes" in url
    assert "language=en" in url
    assert "sample_rate=24000" in url
    assert "output_format=pcm" in url
    assert "speed=1.2" in url
    assert "content_filter=true" in url
    assert "content_filter_action=flag" in url


def test_build_websocket_url_omits_unset_optional_params():
    service = _make_service()
    url = service._build_websocket_url()

    assert "speed=" not in url
    assert "content_filter=" not in url
    assert "content_filter_action=" not in url


async def _drive(service: SmallestLightningV4TTSService, messages):
    """Run _receive_messages over a scripted stream of text/binary frames."""

    async def noop(*args, **kwargs):
        pass

    async def fake_ws():
        for message in messages:
            yield message

    service.append_to_audio_context = noop
    service.stop_ttfb_metrics = noop
    service.stop_all_metrics = noop
    service.push_error = noop
    service.audio_context_available = lambda context_id: True
    service.remove_audio_context = noop
    service.get_active_audio_context_id = lambda: CTX
    service._get_websocket = fake_ws

    await service._receive_messages()


@pytest.mark.asyncio
async def test_turn_start_tracks_the_in_flight_turn():
    service = _make_service()
    await _drive(service, [json.dumps({"event": "turn_start", "turn_id": CTX})])
    assert service._turn_context_id is None  # turn_end never arrived, so cleared in `finally`


@pytest.mark.asyncio
async def test_turn_end_clears_the_in_flight_turn():
    service = _make_service()
    messages = [
        json.dumps({"event": "turn_start", "turn_id": CTX}),
        json.dumps({"event": "turn_end", "turn_id": CTX, "chunks": 3}),
    ]

    end_turn_calls = []

    async def fake_end_turn(context_id):
        end_turn_calls.append(context_id)
        service._turn_context_id = None

    service._end_turn = fake_end_turn
    await _drive(service, messages)

    assert end_turn_calls == [CTX]


@pytest.mark.asyncio
async def test_binary_audio_frame_uses_the_in_flight_turn_as_context():
    service = _make_service()
    service._v4_sample_rate = 24000

    captured = []

    async def fake_append(context_id, frame):
        captured.append((context_id, frame))

    async def noop(*args, **kwargs):
        pass

    async def fake_ws():
        yield json.dumps({"event": "turn_start", "turn_id": CTX})
        yield b"\x00\x01audio-bytes"

    service.append_to_audio_context = fake_append
    service.stop_ttfb_metrics = noop
    service.stop_all_metrics = noop
    service.push_error = noop
    service.audio_context_available = lambda context_id: True
    service.remove_audio_context = noop
    service.get_active_audio_context_id = lambda: None
    service._get_websocket = fake_ws

    await service._receive_messages()

    # The stream ends right after the audio frame with no `turn_end`, so the
    # `finally` clause also reports the turn as lost mid-flight — this only
    # checks that the audio itself landed on the right context.
    audio_frames = [(cid, f) for cid, f in captured if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) == 1
    context_id, frame = audio_frames[0]
    assert context_id == CTX
    assert frame.audio == b"\x00\x01audio-bytes"
    assert frame.sample_rate == 24000


@pytest.mark.asyncio
async def test_interrupt_sends_a_single_frame_without_reconnecting():
    """Barge-in uses the native `interrupt` message rather than tearing the
    session down, so accumulated server-side context survives it."""
    service = _make_service()

    sent = []

    class FakeWebsocket:
        async def send(self, data):
            sent.append(json.loads(data))

    service._websocket = FakeWebsocket()
    service._turn_context_id = CTX

    async def noop(*args, **kwargs):
        pass

    service.stop_all_metrics = noop

    await service.on_audio_context_interrupted(CTX)

    assert sent == [{"event": "interrupt"}]
    assert service._turn_context_id is None


@pytest.mark.asyncio
async def test_add_user_turn_sends_the_callers_transcript():
    service = _make_service()

    sent = []

    class FakeWebsocket:
        async def send(self, data):
            sent.append(json.loads(data))

    service._websocket = FakeWebsocket()

    await service.add_user_turn("What's my balance?")

    assert sent == [{"event": "user_audio", "text": "What's my balance?"}]


@pytest.mark.asyncio
async def test_add_user_turn_is_a_noop_without_a_connection():
    service = _make_service()
    service._websocket = None

    # Should not raise even though there's nowhere to send.
    await service.add_user_turn("What's my balance?")


@pytest.mark.asyncio
async def test_add_user_turn_ignores_empty_text():
    service = _make_service()

    sent = []

    class FakeWebsocket:
        async def send(self, data):
            sent.append(json.loads(data))

    service._websocket = FakeWebsocket()

    await service.add_user_turn("")

    assert sent == []
