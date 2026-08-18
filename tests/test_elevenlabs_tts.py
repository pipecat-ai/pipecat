#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ElevenLabs TTS alignment handling."""

import asyncio
import base64
import json
import unittest
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close
from websockets.protocol import State

from pipecat.services.elevenlabs.tts import (
    _KEEPALIVE_CONTEXT_ID,
    ElevenLabsDialogueTTSService,
    ElevenLabsHttpTTSService,
    ElevenLabsTTSService,
    _DialogueContext,
    _normalize_ttd_alignment,
    _select_alignment,
    _strip_utterance_leading_spaces,
    _word_timestamps_include_inter_frame_spaces,
    calculate_word_times,
)
from pipecat.services.tts_service import TextAggregationMode
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

_WS_ALIGNMENT_KEYS = ("chars", "charStartTimesMs", "charDurationsMs")


def _chunk(text: str) -> dict[str, list[Any]]:
    chars = list(text)
    return {
        "chars": chars,
        "charStartTimesMs": [i * 100 for i in range(len(chars))],
        "charDurationsMs": [100 for _ in chars],
    }


def _words_from_chunks(chunks: list[dict[str, list[Any]]]) -> list[str]:
    cumulative_time = 0.0
    partial_word = ""
    partial_word_start_time = 0.0
    word_times = []
    alignment_started = False

    for chunk in chunks:
        alignment = _strip_utterance_leading_spaces(
            chunk,
            _WS_ALIGNMENT_KEYS,
            not alignment_started,
        )
        alignment_started = True
        chunk_word_times, partial_word, partial_word_start_time = calculate_word_times(
            alignment,
            cumulative_time,
            partial_word,
            partial_word_start_time,
        )
        word_times.extend(chunk_word_times)

        starts = alignment["charStartTimesMs"]
        durations = alignment["charDurationsMs"]
        if starts and durations:
            cumulative_time += (starts[-1] + durations[-1]) / 1000.0

    if partial_word:
        word_times.append((partial_word, partial_word_start_time))

    return [word for word, _ in word_times]


def _concatenate_words_for_language(words: list[str], language: str) -> str:
    includes_inter_frame_spaces = _word_timestamps_include_inter_frame_spaces(language)
    return concatenate_aggregated_text(
        [
            TextPartForConcatenation(
                word,
                includes_inter_part_spaces=includes_inter_frame_spaces,
            )
            for word in words
        ]
    )


def test_elevenlabs_flash_alignment_preserves_inter_word_chunk_space():
    chunks = [
        _chunk(" Why did the math book"),
        _chunk(" look so sad? "),
        _chunk(" Because it had too m"),
        _chunk("any problems. "),
    ]

    assert _words_from_chunks(chunks) == [
        "Why",
        "did",
        "the",
        "math",
        "book",
        "look",
        "so",
        "sad?",
        "Because",
        "it",
        "had",
        "too",
        "many",
        "problems.",
    ]


def test_elevenlabs_japanese_timestamp_chunks_reassemble_without_spaces():
    words = _words_from_chunks(
        [
            _chunk("どんなことでも気 "),
            _chunk("軽に相談してくださいね。 "),
        ]
    )

    assert words == ["どんなことでも気", "軽に相談してくださいね。"]
    assert (
        _concatenate_words_for_language(words, language="ja")
        == "どんなことでも気軽に相談してくださいね。"
    )


def test_elevenlabs_chinese_timestamp_chunks_reassemble_without_spaces():
    words = _words_from_chunks(
        [
            _chunk("你好，我是 "),
            _chunk("你的智能助手。 "),
        ]
    )

    assert words == ["你好，我是", "你的智能助手。"]
    assert _concatenate_words_for_language(words, language="zh-CN") == "你好，我是你的智能助手。"


def test_elevenlabs_english_timestamp_chunks_reassemble_with_spaces():
    words = ["Hello", "world."]

    assert _concatenate_words_for_language(words, language="en") == "Hello world."


def test_elevenlabs_timestamp_spacing_languages():
    assert _word_timestamps_include_inter_frame_spaces("ja") is True
    assert _word_timestamps_include_inter_frame_spaces("zh-CN") is True
    assert _word_timestamps_include_inter_frame_spaces("en") is False


def test_elevenlabs_alignment_strips_only_utterance_leading_spaces():
    first = _strip_utterance_leading_spaces(_chunk("  Hello"), _WS_ALIGNMENT_KEYS, True)
    subsequent = _strip_utterance_leading_spaces(_chunk(" world"), _WS_ALIGNMENT_KEYS, False)

    assert first["chars"] == list("Hello")
    assert subsequent["chars"] == list(" world")


def test_select_alignment_default_prefers_alignment():
    msg = {
        "alignment": _chunk("Hello"),
        "normalizedAlignment": _chunk(" Hello"),
    }
    selected = _select_alignment(
        msg,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=False,
    )
    assert selected is not None
    assert selected["chars"] == list("Hello")


def test_select_alignment_dictionary_mode_prefers_normalized():
    msg = {
        "alignment": _chunk("Hello"),
        "normalizedAlignment": _chunk(" Hello"),
    }
    selected = _select_alignment(
        msg,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=True,
    )
    assert selected is not None
    assert selected["chars"] == list(" Hello")


def test_select_alignment_falls_back_when_preferred_missing():
    msg_default = {"normalizedAlignment": _chunk(" Hello")}
    selected = _select_alignment(
        msg_default,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=False,
    )
    assert selected is not None
    assert selected["chars"] == list(" Hello")

    msg_dict = {"alignment": _chunk("Hello")}
    selected = _select_alignment(
        msg_dict,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=True,
    )
    assert selected is not None
    assert selected["chars"] == list("Hello")


def test_select_alignment_falls_back_when_preferred_null():
    msg = {"alignment": None, "normalizedAlignment": _chunk(" Hello")}
    selected = _select_alignment(
        msg,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=False,
    )
    assert selected is not None
    assert selected["chars"] == list(" Hello")


def test_select_alignment_returns_none_when_both_missing():
    assert (
        _select_alignment(
            {},
            normalized_key="normalizedAlignment",
            alignment_key="alignment",
            prefer_normalized=False,
        )
        is None
    )
    assert (
        _select_alignment(
            {"alignment": None, "normalizedAlignment": None},
            normalized_key="normalizedAlignment",
            alignment_key="alignment",
            prefer_normalized=True,
        )
        is None
    )


def test_select_alignment_works_with_http_field_names():
    msg = {
        "alignment": {"characters": list("Hi")},
        "normalized_alignment": {"characters": list(" Hi")},
    }
    selected = _select_alignment(
        msg,
        normalized_key="normalized_alignment",
        alignment_key="alignment",
        prefer_normalized=False,
    )
    assert selected is not None
    assert selected["characters"] == list("Hi")

    selected = _select_alignment(
        msg,
        normalized_key="normalized_alignment",
        alignment_key="alignment",
        prefer_normalized=True,
    )
    assert selected is not None
    assert selected["characters"] == list(" Hi")


# ---------------------------------------------------------------------------
# Keepalive vs context-init race
#
# The keepalive must only stamp a context_id once its context-init (carrying
# voice_settings) has been sent. Stamping it earlier makes the keepalive the
# context's first message, with no voice_settings, and ElevenLabs rejects the
# later context-init with a 1008 policy violation.
# ---------------------------------------------------------------------------


class _FakeWebSocket:
    """Minimal stand-in for the ElevenLabs websocket that records sends."""

    def __init__(self):
        self.state = State.OPEN
        self.sent: list[dict] = []

    async def send(self, data: str):
        self.sent.append(json.loads(data))


def _make_service() -> ElevenLabsTTSService:
    return ElevenLabsTTSService(
        api_key="test-key",
        settings=ElevenLabsTTSService.Settings(
            voice="test-voice",
            stability=0.55,
            similarity_boost=0.85,
            use_speaker_boost=True,
            speed=0.81,
        ),
    )


@pytest.mark.asyncio
async def test_keepalive_does_not_stamp_context_before_init():
    """During the pre-init window the keepalive must not stamp the new context_id."""
    service = _make_service()
    ws = _FakeWebSocket()
    service._websocket = ws

    # Simulate the start of an LLM turn: TTSService sets the turn context id on
    # LLMFullResponseStartFrame, before run_tts sends the voice_settings init.
    service._turn_context_id = "ctx-1"
    service._playing_context_id = None
    assert "ctx-1" not in service._context_init_sent

    await service._send_keepalive()

    # Context-less keepalive: the real context-init stays the context's first
    # message, so ElevenLabs won't reject it with 1008.
    assert ws.sent == [{"text": ""}]


@pytest.mark.asyncio
async def test_keepalive_stamps_context_after_init():
    """Once the context-init has been sent, the keepalive targets that context."""
    service = _make_service()
    ws = _FakeWebSocket()
    service._websocket = ws
    service._turn_context_id = "ctx-1"
    service._playing_context_id = None
    # run_tts records the context once its voice_settings init has gone out.
    service._context_init_sent.add("ctx-1")

    await service._send_keepalive()

    assert ws.sent == [{"text": "", "context_id": "ctx-1"}]


@pytest.mark.asyncio
async def test_keepalive_without_active_context_sends_empty():
    """With no active context, the keepalive sends a plain empty message."""
    service = _make_service()
    ws = _FakeWebSocket()
    service._websocket = ws
    service._turn_context_id = None
    service._playing_context_id = None

    await service._send_keepalive()

    assert ws.sent == [{"text": ""}]


class _FakeHttpResponse:
    """Minimal aiohttp response stand-in; the 400 makes run_tts bail after posting."""

    status = 400

    async def text(self):
        return "rejected"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeHttpSession:
    """Records the JSON payload of each POST."""

    def __init__(self):
        self.payloads: list[dict] = []

    def post(self, url, json=None, headers=None, params=None):
        self.payloads.append(json)
        return _FakeHttpResponse()


async def _http_payload_for_model(model: str) -> dict:
    session = _FakeHttpSession()
    service = ElevenLabsHttpTTSService(
        api_key="test-key",
        aiohttp_session=session,
        settings=ElevenLabsHttpTTSService.Settings(voice="test-voice", model=model),
    )
    service._previous_text = "Hello!"
    async for _ in service.run_tts("How can I assist you today?", "ctx-1"):
        pass
    return session.payloads[0]


@pytest.mark.asyncio
async def test_http_payload_includes_previous_text_when_supported():
    payload = await _http_payload_for_model("eleven_flash_v2_5")
    assert payload["previous_text"] == "Hello!"


@pytest.mark.asyncio
async def test_http_payload_omits_previous_text_for_eleven_v3():
    payload = await _http_payload_for_model("eleven_v3")
    assert "previous_text" not in payload


# ---------------------------------------------------------------------------
# Disconnect vs server-initiated close race
#
# When the server closes the websocket first (normal during teardown), the
# close-handshake send in _disconnect_websocket raises ConnectionClosed. That
# must not be reported as a pipeline error: a non-fatal ErrorFrame here can
# e.g. trigger a spurious ServiceSwitcherStrategyFailover switch on shutdown.
# ---------------------------------------------------------------------------


class _ClosedWebSocket:
    """Websocket stand-in whose sends fail with a normal close."""

    state = State.OPEN

    async def send(self, data: str):
        raise ConnectionClosedOK(Close(1001, "going away"), Close(1001, "going away"), True)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_disconnect_does_not_push_error_when_server_closed_first():
    """A ConnectionClosed during the disconnect handshake is not a pipeline error."""
    service = _make_service()
    service._websocket = _ClosedWebSocket()

    errors = []

    async def push_error(error_msg=None, exception=None):
        errors.append(error_msg)

    service.push_error = push_error

    await service._disconnect_websocket()

    assert errors == []
    assert service._websocket is None


# ---------------------------------------------------------------------------
# Text-to-Dialogue
# ---------------------------------------------------------------------------


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
async def test_dialogue_flush_skipped_when_nothing_new_was_sent():
    """Batches sent and batches acknowledged have to stay in step."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    await service.flush_audio("ctx-1")
    assert ws.sent == []

    await service._send_text("Hello there.", "ctx-1")
    await service.flush_audio("ctx-1")
    await service.flush_audio("ctx-1")

    assert ws.sent.count({"context_id": "ctx-1", "flush": True}) == 1
    assert service._contexts["ctx-1"].outstanding_batches == 1


@pytest.mark.asyncio
async def test_dialogue_turn_end_waits_for_every_flushed_batch():
    """A close discards batches that haven't started, truncating the turn."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    for sentence in ("First sentence.", "Second sentence."):
        await service._send_text(sentence, "ctx-1")
        await service.flush_audio("ctx-1")
    assert service._contexts["ctx-1"].outstanding_batches == 2

    service._turn_context_id = "ctx-1"
    await service.on_turn_context_completed()

    def closes():
        return ws.sent.count({"context_id": "ctx-1", "close_context": True})

    assert closes() == 0, "closed before any batch was acknowledged"
    assert service._contexts["ctx-1"].close_when_drained is True

    await service._handle_message({"context_id": "ctx-1", "is_final_audio_for_turn": True})
    assert closes() == 0, "closed while a batch was still outstanding"

    await service._handle_message({"context_id": "ctx-1", "is_final_audio_for_turn": True})
    assert closes() == 1


@pytest.mark.asyncio
async def test_dialogue_turn_end_closes_once_batches_already_acknowledged():
    """Generation can finish before the turn does; then the close is immediate."""
    service = _make_dialogue_service()
    ws = _FakeWebSocket()
    await _open_dialogue_context(service, ws)

    await service._send_text("Sure.", "ctx-1")
    await service.flush_audio("ctx-1")
    await service._handle_message({"context_id": "ctx-1", "is_final_audio_for_turn": True})

    service._turn_context_id = "ctx-1"
    await service.on_turn_context_completed()

    assert ws.sent[-1] == {"context_id": "ctx-1", "close_context": True}


@pytest.mark.asyncio
async def test_dialogue_turn_final_does_not_end_the_audio_context():
    """Turn finals arrive per batch; only is_final ends a context."""
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
