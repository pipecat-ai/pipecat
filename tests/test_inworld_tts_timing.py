#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for InworldTTSService generation-timing scoping."""

import json

import pytest

from pipecat.services.inworld.tts import InworldTTSService

LIVE_CTX = "ctx-live"
CLOSED_CTX = "ctx-closed"


def _alignment_msg(context_id: str | None, words, starts, ends) -> str:
    """A timestamp-only audioChunk, as the ASYNC strategy delivers alignment."""
    result = {
        "audioChunk": {
            "timestampInfo": {
                "wordAlignment": {
                    "words": words,
                    "wordStartTimeSeconds": starts,
                    "wordEndTimeSeconds": ends,
                }
            }
        }
    }
    if context_id is not None:
        result["contextId"] = context_id
    return json.dumps({"result": result})


def _flush_msg(context_id: str | None) -> str:
    result = {"flushCompleted": {}}
    if context_id is not None:
        result["contextId"] = context_id
    return json.dumps({"result": result})


def _make_service() -> InworldTTSService:
    return InworldTTSService(api_key="test-key")


async def _drive(service: InworldTTSService, messages):
    """Run _receive_messages over a scripted stream, capturing word timestamps.

    Only ``LIVE_CTX`` has an audio context, and the capture replaces
    ``add_word_timestamps``, which would itself drop words for a dead context.
    """
    captured = []

    async def fake_add_word_timestamps(word_times, context_id=None, **kwargs):
        captured.extend(word_times)

    async def fake_ws():
        for message in messages:
            yield message

    service.add_word_timestamps = fake_add_word_timestamps
    service.audio_context_available = lambda context_id: context_id == LIVE_CTX
    service._get_websocket = fake_ws

    await service._receive_messages()
    return captured


@pytest.mark.asyncio
async def test_closed_context_timing_does_not_shift_live_context():
    """A closed context's trailing alignment and flush leave the live context alone."""
    service = _make_service()

    messages = [
        _alignment_msg(CLOSED_CTX, ["stale"], [0.1], [10.75]),
        _flush_msg(CLOSED_CTX),
        _alignment_msg(LIVE_CTX, ["hello"], [0.25], [0.9]),
    ]
    captured = await _drive(service, messages)

    assert captured == [("hello", pytest.approx(0.25))]


@pytest.mark.asyncio
async def test_closed_context_flush_does_not_advance_live_generation():
    """A closed context's flush cannot promote the live context's partial end time."""
    service = _make_service()

    # The flush lands between two alignments of one live generation, when
    # _generation_end_time holds that generation's end so far.
    messages = [
        _alignment_msg(LIVE_CTX, ["one", "two"], [0.0, 0.5], [0.4, 0.7]),
        _flush_msg(CLOSED_CTX),
        _alignment_msg(LIVE_CTX, ["three"], [0.8], [1.2]),
    ]
    captured = await _drive(service, messages)

    assert captured == [
        ("one", pytest.approx(0.0)),
        ("two", pytest.approx(0.5)),
        ("three", pytest.approx(0.8)),
    ]
    assert service._cumulative_time == 0.0


@pytest.mark.asyncio
async def test_generations_within_one_context_accumulate():
    """Raw times restart each generation, so a context's own flush advances its offset."""
    service = _make_service()

    messages = [
        _alignment_msg(LIVE_CTX, ["one", "two"], [0.0, 0.5], [0.4, 0.9]),
        _flush_msg(LIVE_CTX),
        _alignment_msg(LIVE_CTX, ["three"], [0.1], [0.6]),
    ]
    captured = await _drive(service, messages)

    assert captured == [
        ("one", pytest.approx(0.0)),
        ("two", pytest.approx(0.5)),
        ("three", pytest.approx(1.0)),
    ]


@pytest.mark.asyncio
async def test_messages_without_a_context_id_still_advance_timing():
    """A response carrying no contextId cannot be attributed, so timing still applies."""
    service = _make_service()

    messages = [
        _alignment_msg(None, ["one"], [0.0], [0.9]),
        _flush_msg(None),
        _alignment_msg(None, ["two"], [0.1], [0.6]),
    ]
    captured = await _drive(service, messages)

    assert captured == [("one", pytest.approx(0.0)), ("two", pytest.approx(1.0))]
