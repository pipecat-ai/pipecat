#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SmallestTTSService behavior."""

import json

import pytest

from pipecat.services.smallest.tts import SmallestTTSModel, SmallestTTSService, SmallestTTSSettings
from pipecat.transcriptions.language import Language

CTX = "ctx-1"


def _word_msg(word: str, start: float, end: float, request_id: str, word_id: int = 0) -> str:
    return json.dumps(
        {
            "request_id": request_id,
            "status": "word_timestamp",
            "data": {"id": word_id, "word": word, "start": start, "end": end},
        }
    )


def _make_service() -> SmallestTTSService:
    return SmallestTTSService(api_key="test-key")


def test_smallest_tts_uses_live_endpoint_and_payload_model():
    service = SmallestTTSService(
        api_key="test-key",
        settings=SmallestTTSSettings(
            model=SmallestTTSModel.LIGHTNING_V3_1.value,
            voice="sophia",
            language=Language.EN,
            speed=1.2,
        ),
    )

    assert service._build_websocket_url() == "wss://api.smallest.ai/waves/v1/tts/live"

    message = service._build_msg("hello")
    assert message["text"] == "hello"
    assert message["voice_id"] == "sophia"
    assert message["model"] == "lightning_v3.1"
    assert message["language"] == "en"
    assert message["speed"] == 1.2
    assert message["word_timestamps"] is True
    assert message["output_format"] == "pcm"


def test_smallest_tts_defaults_to_pro_model_voice_pair():
    service = SmallestTTSService(api_key="test-key")

    assert service._settings.model == "lightning_v3.1_pro"
    assert service._settings.voice == "meher"


def test_word_timestamps_enabled_by_default():
    """Word timestamps are on by default and drive text-frame emission."""
    service = SmallestTTSService(api_key="test-key")
    assert service._word_timestamps is True
    # Word events produce the TTSTextFrames, so the base must not push whole text.
    assert service._push_text_frames is False
    assert service._build_msg("hi")["word_timestamps"] is True


def test_word_timestamps_disabled_pushes_whole_text():
    """Disabling word timestamps flips the service back to whole-text frames."""
    service = SmallestTTSService(api_key="test-key", word_timestamps=False)
    assert service._word_timestamps is False
    assert service._push_text_frames is True
    assert "word_timestamps" not in service._build_msg("hi")


async def _drive(service: SmallestTTSService, messages):
    """Run _receive_messages over a scripted stream, capturing word timestamps."""
    captured = []

    async def fake_add_word_timestamps(word_times, context_id=None, **kwargs):
        captured.extend(word_times)

    async def noop(*args, **kwargs):
        pass

    async def fake_ws():
        for message in messages:
            yield message

    service.add_word_timestamps = fake_add_word_timestamps
    service.append_to_audio_context = noop
    service.stop_ttfb_metrics = noop
    service.stop_all_metrics = noop
    service.get_active_audio_context_id = lambda: CTX
    service._get_websocket = fake_ws

    await service._receive_messages()
    return captured


@pytest.mark.asyncio
async def test_word_timestamps_offset_across_requests():
    """Later requests in a turn are shifted onto the turn's playback timeline.

    Smallest reports per-request timestamps that reset to ~0 each request and
    only emits one ``complete`` for the whole turn, so the request boundary is
    detected by a change in ``request_id``. The second request's words must be
    offset by the prior request's last-word ``end``.
    """
    service = _make_service()

    # Request A (id "a"): word at 0.2s, ending at 0.5s.
    # Request B (id "b", same turn): word at 0.1s -> 0.1 + 0.5 = 0.6s.
    messages = [
        _word_msg("Hello", 0.2, 0.5, request_id="a"),
        _word_msg("World", 0.1, 0.4, request_id="b"),
    ]
    captured = await _drive(service, messages)

    assert captured == [("Hello", pytest.approx(0.2)), ("World", pytest.approx(0.6))]


@pytest.mark.asyncio
async def test_offset_accumulates_across_multiple_requests():
    """The offset compounds across three sequential requests in one turn."""
    service = _make_service()

    messages = [
        _word_msg("one", 0.0, 1.0, request_id="a"),
        _word_msg("two", 0.0, 2.0, request_id="b"),  # offset by 1.0
        _word_msg("three", 0.5, 1.0, request_id="c"),  # offset by 1.0 + 2.0
    ]
    captured = await _drive(service, messages)

    assert captured == [
        ("one", pytest.approx(0.0)),
        ("two", pytest.approx(1.0)),
        ("three", pytest.approx(3.5)),
    ]


@pytest.mark.asyncio
async def test_multiple_words_in_one_request_share_offset():
    """All words within a request use the same offset; only `end` grows."""
    service = _make_service()

    messages = [
        _word_msg("a", 0.0, 0.4, request_id="r1", word_id=0),
        _word_msg("b", 0.4, 0.9, request_id="r1", word_id=1),
        _word_msg("c", 0.1, 0.5, request_id="r2", word_id=0),  # offset by 0.9
    ]
    captured = await _drive(service, messages)

    assert captured == [
        ("a", pytest.approx(0.0)),
        ("b", pytest.approx(0.4)),
        ("c", pytest.approx(1.0)),
    ]


@pytest.mark.asyncio
async def test_word_timestamp_offset_resets_on_new_turn():
    """on_turn_context_created (a new LLM turn) clears the accumulated offset."""
    service = _make_service()

    # First turn: two requests, so the offset accumulates to 0.5.
    await _drive(
        service,
        [
            _word_msg("Hello", 0.2, 0.5, request_id="a"),
            _word_msg("World", 0.1, 0.4, request_id="b"),
        ],
    )
    assert service._cumulative_time == pytest.approx(0.5)

    # A new turn resets the timeline.
    await service.on_turn_context_created("ctx-2")
    assert service._cumulative_time == 0.0
    assert service._wt_request_id is None

    captured = await _drive(service, [_word_msg("Fresh", 0.3, 0.6, request_id="c")])
    assert captured == [("Fresh", pytest.approx(0.3))]
