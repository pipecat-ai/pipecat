#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ElevenLabs Text-to-Dialogue TTS service."""

import asyncio
import base64
import json
import unittest

import pytest
from websockets.protocol import State

from pipecat.services.elevenlabs.dialogue.tts import (
    _KEEPALIVE_CONTEXT_ID,
    ElevenLabsDialogueTTSService,
    _DialogueContext,
    _normalize_ttd_alignment,
)
from pipecat.services.elevenlabs.tts import calculate_word_times
from pipecat.services.tts_service import TextAggregationMode

_WS_ALIGNMENT_KEYS = ("chars", "charStartTimesMs", "charDurationsMs")


class _FakeWebSocket:
    """Minimal stand-in for the ElevenLabs websocket that records sends."""

    def __init__(self):
        self.state = State.OPEN
        self.sent: list[dict] = []

    async def send(self, data: str):
        self.sent.append(json.loads(data))


def _make_dialogue_service(**settings_kwargs) -> ElevenLabsDialogueTTSService:
    settings = ElevenLabsDialogueTTSService.Settings(voice="test-voice", **settings_kwargs)
    return ElevenLabsDialogueTTSService(api_key="test-key", settings=settings)


#: One 24kHz PCM frame of silence, base64-encoded, as the server sends audio.
_SILENCE_B64 = base64.b64encode(b"\x00" * 480).decode()


async def _open_dialogue_context(service, ws, context_id="ctx-1"):
    """Put the service in the state run_tts leaves behind for an open context."""
    service._websocket = ws
    service._contexts[context_id] = _DialogueContext()
    service._audio_contexts[context_id] = asyncio.Queue()


@pytest.mark.asyncio
async def test_dialogue_text_is_sent_as_inputs():
    """Text goes out as an inputs array tagged with the configured voice."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    await service._send_text("Hello there.", "ctx-1")

    assert ws.sent == [
        {
            "context_id": "ctx-1",
            "inputs": [{"text": "Hello there.", "voice_id": "test-voice", "new_turn": True}],
        }
    ]


@pytest.mark.asyncio
async def test_dialogue_new_turn_only_on_first_input():
    """new_turn resets prosody once per context, not on every input."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    await service._send_text("First sentence.", "ctx-1")
    await service._send_text("Second sentence.", "ctx-1")

    assert [m["inputs"][0]["new_turn"] for m in ws.sent] == [True, False]


@pytest.mark.asyncio
async def test_dialogue_text_for_unregistered_context_is_dropped():
    """Messages naming an unregistered context would close the socket with 1008."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    service._websocket = ws

    await service._send_text("Hello there.", "ctx-unknown")
    await service.flush_audio("ctx-unknown")

    assert ws.sent == []


@pytest.mark.asyncio
async def test_dialogue_interruption_closes_context_without_reconnect():
    """Interruptions cancel via close_context on the open socket."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    disconnects = []
    service._disconnect = lambda: disconnects.append(True)

    await service._close_context("ctx-1")

    assert ws.sent == [{"context_id": "ctx-1", "close_context": True}]
    assert disconnects == []
    assert service._contexts["ctx-1"].registered is False


@pytest.mark.asyncio
async def test_dialogue_close_context_is_idempotent():
    """A context already closed by the server isn't closed twice."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    await service._close_context("ctx-1")
    await service._close_context("ctx-1")

    assert len(ws.sent) == 1


@pytest.mark.asyncio
async def test_dialogue_keepalive_context_is_registered_on_connect():
    """The connection idles out in 20s without a context to keep alive."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    service._websocket = ws

    await service._register_keepalive_context()

    keepalive_id = _KEEPALIVE_CONTEXT_ID
    assert ws.sent == [{"context_id": keepalive_id, "voices": ["test-voice"]}]
    assert keepalive_id in service._contexts


@pytest.mark.asyncio
async def test_dialogue_keepalive_targets_its_own_context():
    """A keep_alive without a registered context_id is rejected with 1008."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    service._websocket = ws
    await service._register_keepalive_context()
    ws.sent.clear()

    await service._send_keepalive()

    assert ws.sent == [{"context_id": _KEEPALIVE_CONTEXT_ID, "keep_alive": True}]


@pytest.mark.asyncio
async def test_dialogue_keepalive_silent_before_registration():
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    service._websocket = ws

    await service._send_keepalive()

    assert ws.sent == []


@pytest.mark.asyncio
async def test_dialogue_keepalive_context_messages_are_ignored():
    """Nothing about the keepalive context should reach the audio pipeline."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    service._websocket = ws

    appended = []
    service.append_to_audio_context = lambda ctx, frame: appended.append((ctx, frame))

    await service._handle_message({"context_id": _KEEPALIVE_CONTEXT_ID, "is_final": True})

    assert appended == []


@pytest.mark.asyncio
async def test_dialogue_flush_targets_registered_contexts_only():
    """The server rejects any message naming a context it has closed."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    await service.flush_audio("ctx-1")
    assert ws.sent == [{"context_id": "ctx-1", "flush": True}]

    service._contexts["ctx-1"].registered = False
    await service.flush_audio("ctx-1")

    assert ws.sent.count({"context_id": "ctx-1", "flush": True}) == 1


@pytest.mark.asyncio
async def test_dialogue_turn_end_closes_the_context():
    """Ending a turn sends the close, which is what generates the turn's tail."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    await service._send_text("Sure.", "ctx-1")

    service._turn_context_id = "ctx-1"
    await service.on_turn_context_completed()

    assert ws.sent[-1] == {"context_id": "ctx-1", "close_context": True}


@pytest.mark.asyncio
async def test_dialogue_turn_final_does_not_end_the_audio_context():
    """Turn finals arrive per generation batch; only is_final ends a context."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    appended = []
    service.append_to_audio_context = lambda ctx, frame: appended.append(frame)

    await service._handle_message({"context_id": "ctx-1", "is_final_audio_for_turn": True})

    assert appended == []


@pytest.mark.asyncio
async def test_dialogue_interruption_closes_without_waiting_for_audio():
    """Before generation starts, an immediate close is what cancels the text."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    await service._close_context("ctx-1")

    assert ws.sent == [{"context_id": "ctx-1", "close_context": True}]


def test_dialogue_stability_is_passed_through():
    """The API validates the range itself; it accepts any value in it."""
    for given in (0.0, 0.25, 0.3, 0.5, 0.7, 1.0):
        service = _make_dialogue_service(stability=given)
        assert service._voice_settings == {"stability": given}


def test_dialogue_voice_settings_carries_stability_only():
    """Text-to-Dialogue ignores the other voice settings entirely."""
    service = _make_dialogue_service(stability=0.3)
    assert set(service._voice_settings) == {"stability"}


def test_dialogue_voice_settings_omitted_when_stability_unset():
    service = _make_dialogue_service()
    assert service._voice_settings is None


def test_dialogue_voice_change_does_not_require_reconnect():
    """Voices are registered per context, so a voice change just needs a new one."""
    assert "voice" not in ElevenLabsDialogueTTSService.Settings.URL_FIELDS
    assert "voice" in ElevenLabsDialogueTTSService.Settings.VOICE_SETTINGS_FIELDS


def test_dialogue_alignment_is_normalized_for_word_times():
    """Text-to-Dialogue sends snake_case alignment; shared helpers expect camelCase."""
    normalized = _normalize_ttd_alignment(
        {
            "chars": ["H", "i", " ", "t", "h", "e", "r", "e"],
            "char_start_times_ms": [0, 50, 100, 120, 170, 220, 270, 320],
            "char_durations_ms": [50, 50, 20, 50, 50, 50, 50, 50],
        }
    )

    assert set(normalized) == set(_WS_ALIGNMENT_KEYS)

    word_times, partial, _ = calculate_word_times(normalized, 0.0, "", 0.0)
    assert [word for word, _ in word_times] == ["Hi"]
    assert partial == "there"


@pytest.mark.asyncio
async def test_dialogue_wordless_alignment_still_advances_the_clock():
    """Trailing punctuation arrives on its own; its span must not be dropped."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    async def _noop(*args, **kwargs):
        return None

    service.add_word_timestamps = _noop

    # "Hi there" -- completes "Hi", carries "there".
    await service._handle_message(
        {
            "context_id": "ctx-1",
            "alignment": {
                "chars": ["H", "i", " ", "t", "h", "e", "r", "e"],
                "char_start_times_ms": [0, 120, 240, 293, 347, 400, 453, 507],
                "char_durations_ms": [120, 120, 53, 54, 53, 53, 54, 53],
            },
        }
    )
    after_first = service._cumulative_time
    assert after_first == pytest.approx(0.560)

    # "!" alone: completes no word, but occupies 320ms of audio.
    await service._handle_message(
        {
            "context_id": "ctx-1",
            "alignment": {
                "chars": ["!"],
                "char_start_times_ms": [0],
                "char_durations_ms": [320],
            },
        }
    )

    assert service._cumulative_time == pytest.approx(after_first + 0.320), (
        "word-less alignment chunk dropped from the clock; later word timestamps would run early"
    )


def test_dialogue_appends_trailing_space_to_inputs():
    """Consecutive inputs are concatenated verbatim by the server."""
    service = _make_dialogue_service()

    assert service._append_trailing_space is True
    assert service._prepare_text_for_tts("Hi there!") == "Hi there! "
    assert service._prepare_text_for_tts("Hi there! ") == "Hi there! "


def test_dialogue_always_aggregates_sentences():
    """Each flush generates independently, so tokens would synthesize out of context."""
    service = _make_dialogue_service()

    assert service._text_aggregation_mode is TextAggregationMode.SENTENCE
    assert service._is_streaming_tokens is False


def test_dialogue_token_aggregation_is_refused():
    """Sentence aggregation holds even when the caller asks for tokens."""
    settings = ElevenLabsDialogueTTSService.Settings(voice="test-voice")
    service = ElevenLabsDialogueTTSService(
        api_key="test-key",
        settings=settings,
        text_aggregation_mode=TextAggregationMode.TOKEN,
    )

    assert service._text_aggregation_mode is TextAggregationMode.SENTENCE
    assert service._is_streaming_tokens is False


def test_dialogue_non_v3_model_warns_without_raising():
    """The wrong-model warning names the service, so it needs a constructed one."""
    service = _make_dialogue_service(model="eleven_flash_v2_5")

    assert service._settings.model == "eleven_flash_v2_5"


def test_dialogue_appends_trailing_space_under_sentence_aggregation():
    """Sentence aggregation is what makes the trailing space safe to append."""
    service = _make_dialogue_service()

    assert service._append_trailing_space is True
    assert service._prepare_text_for_tts("Hi there!") == "Hi there! "


@pytest.mark.asyncio
async def test_dialogue_drained_alignment_does_not_disturb_the_next_turn():
    """An interrupted context keeps streaming; its alignment belongs to nobody."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    async def _noop(*args, **kwargs):
        return None

    service.add_word_timestamps = _noop

    # The next turn is under way with a fresh clock.
    service._cumulative_time = 0.0
    service._partial_word = ""
    service._audio_contexts = {"ctx-2": None}

    # Late alignment from the interrupted context arrives.
    await service._handle_message(
        {
            "context_id": "ctx-1",
            "alignment": {
                "chars": ["s", "t", "a", "l", "e"],
                "char_start_times_ms": [0, 100, 200, 300, 400],
                "char_durations_ms": [100, 100, 100, 100, 100],
            },
        }
    )

    assert service._cumulative_time == 0.0, "drained context advanced the next turn's clock"
    assert service._partial_word == "", "drained context left a partial word behind"


if __name__ == "__main__":
    unittest.main()
