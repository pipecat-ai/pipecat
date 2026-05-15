#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ElevenLabs TTS alignment handling."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.services.elevenlabs.tts import (
    ElevenLabsTTSService,
    _select_alignment,
    _strip_utterance_leading_spaces,
    calculate_word_times,
)

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


def _init_messages(send_mock: AsyncMock) -> list[dict[str, Any]]:
    """Return the context-init messages (those with text=' ') sent over the WS."""
    return [
        json.loads(call.args[0])
        for call in send_mock.await_args_list
        if json.loads(call.args[0]).get("text") == " "
    ]


def _make_service_with_voice_settings(
    voice_settings: dict[str, float | bool] | None,
) -> ElevenLabsTTSService:
    """Build a service with voice_settings preset and mocked context bookkeeping."""
    service = ElevenLabsTTSService.__new__(ElevenLabsTTSService)
    service._name = "ElevenLabsTTSService"
    service._voice_settings = voice_settings
    service._voice_settings_sent_on_current_ws = False
    service._pronunciation_dictionary_locators = None
    service._cumulative_time = 0
    service._partial_word = ""
    service._partial_word_start_time = 0.0
    service.audio_context_available = MagicMock(return_value=False)
    service.create_audio_context = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.start_tts_usage_metrics = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_voice_settings_only_in_first_context_init():
    """voice_settings must only ride on the first context init per WS connection.

    ElevenLabs' multi-context WebSocket protocol rejects subsequent
    context-init messages that include voice_settings with WS close 1008
    (``voice_settings field must be provided in the first message and then
    either be not provided or not change``).
    """
    service = _make_service_with_voice_settings({"stability": 0.8, "speed": 1.0})
    ws = MagicMock()
    ws.send = AsyncMock()
    service._websocket = ws

    async for _ in service.run_tts("hello", "ctx-1"):
        pass
    async for _ in service.run_tts("world", "ctx-2"):
        pass
    async for _ in service.run_tts("again", "ctx-3"):
        pass

    inits = _init_messages(ws.send)
    assert len(inits) == 3
    assert inits[0].get("voice_settings") == {"stability": 0.8, "speed": 1.0}
    assert "voice_settings" not in inits[1]
    assert "voice_settings" not in inits[2]


@pytest.mark.asyncio
async def test_voice_settings_resent_after_ws_reconnect():
    """A fresh WS connection must (re)send voice_settings on its first context init."""
    service = _make_service_with_voice_settings({"stability": 0.8, "speed": 1.0})
    ws1 = MagicMock()
    ws1.send = AsyncMock()
    service._websocket = ws1

    async for _ in service.run_tts("hello", "ctx-1"):
        pass

    # Simulate _connect_websocket completing on a brand-new WS — the patched
    # _connect_websocket clears the flag at this exact point.
    ws2 = MagicMock()
    ws2.send = AsyncMock()
    service._websocket = ws2
    service._voice_settings_sent_on_current_ws = False

    async for _ in service.run_tts("after reconnect", "ctx-2"):
        pass

    inits_ws2 = _init_messages(ws2.send)
    assert len(inits_ws2) == 1
    assert inits_ws2[0].get("voice_settings") == {"stability": 0.8, "speed": 1.0}


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
