#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the processor that reads each user turn with Pipecat Tone."""

import pytest

from pipecat.expressive.perception import PerceptionSignal
from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    LLMContextAssistantTurnFrame,
    PerceptionMetadataFrame,
    TranscriptionFrame,
)
from pipecat.processors.perception.tone import ToneProcessor
from pipecat.tests.utils import run_test

TAGS = {
    "emotion": {
        "label": "angry",
        "intensity": "low",
        "confidence": 0.71,
        "probabilities": {"neutral": 0.2, "happy": 0.04, "angry": 0.71, "sad": 0.05},
    },
    "move": {"label": "other", "confidence": 0.92, "probabilities": {}},
    "sarcasm": {"sarcastic": True, "probability": 0.99},
    "backchannel": {"is_backchannel": False, "probability": 0.01},
}


class FakeToneModel:
    emotions = ["neutral", "happy", "angry", "sad"]

    def __init__(self, error: Exception | None = None):
        self.reads: list[tuple[str, str | None]] = []
        self._error = error

    def tag(self, text: str, previous: str | None = None):
        self.reads.append((text, previous))
        if self._error:
            raise self._error
        return TAGS


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="now")


@pytest.mark.asyncio
async def test_the_signals_are_declared_at_start():
    down, _ = await run_test(ToneProcessor(model=FakeToneModel()), frames_to_send=[])

    metadata = [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [(t.name, t.labels) for t in metadata[0].perception.signals] == [
        ("emotion", ["neutral", "happy", "angry", "sad"]),
        ("sarcasm", []),
    ]


@pytest.mark.asyncio
async def test_the_reading_is_set_on_its_transcription():
    down, _ = await run_test(
        ToneProcessor(model=FakeToneModel()),
        frames_to_send=[_transcription("Oh great, a forty minute wait.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception == [
        PerceptionSignal("emotion", labels=[("angry", 0.71), ("neutral", 0.2)]),
        PerceptionSignal("sarcasm", value=0.99),
    ]


@pytest.mark.asyncio
async def test_a_turn_is_read_against_the_bots_words_before_it():
    model = FakeToneModel()

    await run_test(
        ToneProcessor(model=model),
        frames_to_send=[
            LLMContextAssistantTurnFrame(
                text="<<calm>> The next visit is in forty minutes. [[sighs]] Sorry.",
                timestamp="now",
            ),
            _transcription("Oh great."),
        ],
    )

    assert model.reads == [("Oh great.", "The next visit is in forty minutes. Sorry.")]


@pytest.mark.asyncio
async def test_interim_transcriptions_are_not_read():
    model = FakeToneModel()

    await run_test(
        ToneProcessor(model=model),
        frames_to_send=[InterimTranscriptionFrame(text="Oh", user_id="", timestamp="now")],
    )

    assert model.reads == []


@pytest.mark.asyncio
async def test_a_failed_read_is_reported_and_the_transcription_still_passes():
    down, up = await run_test(
        ToneProcessor(model=FakeToneModel(error=RuntimeError("no session"))),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert down[1].perception is None
