#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ElevenLabs TTS alignment handling."""

import json
import unittest
from typing import Any

import pytest
from websockets.protocol import State

from pipecat.services.elevenlabs.tts import (
    ElevenLabsHttpTTSService,
    ElevenLabsTTSService,
    _select_alignment,
    _strip_utterance_leading_spaces,
    _word_timestamps_include_inter_frame_spaces,
    calculate_word_times,
)
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


if __name__ == "__main__":
    unittest.main()
