#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for reading a confidence from a Whisper verbose_json transcription."""

import math

from openai.types.audio import TranscriptionVerbose
from openai.types.audio.transcription_segment import TranscriptionSegment

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.whisper.utils import extract_whisper_probability


def _segment(index: int, text: str, avg_logprob: float, num_tokens: int) -> TranscriptionSegment:
    return TranscriptionSegment(
        id=index,
        avg_logprob=avg_logprob,
        compression_ratio=1.0,
        end=float(index + 1),
        no_speech_prob=0.0,
        seek=0,
        start=float(index),
        temperature=0.0,
        text=text,
        tokens=list(range(num_tokens)),
    )


def _frame(segments: list[TranscriptionSegment] | None) -> TranscriptionFrame:
    text = "".join(segment.text for segment in segments or [])
    result = TranscriptionVerbose(duration=2.0, language="english", text=text, segments=segments)
    return TranscriptionFrame(text, "user", "2026-01-01T00:00:00.000Z", result=result)


def test_single_segment():
    frame = _frame([_segment(0, "Hello there.", -0.2, 4)])

    assert math.isclose(extract_whisper_probability(frame), math.exp(-0.2))


def test_every_segment_counts_weighted_by_tokens():
    frame = _frame(
        [
            _segment(0, "Book a table.", -0.1, 3),
            _segment(1, " For four people at seven tonight.", -2.0, 9),
        ]
    )

    expected = math.exp((-0.1 * 3 + -2.0 * 9) / 12)
    assert math.isclose(extract_whisper_probability(frame), expected)


def test_no_segments():
    assert extract_whisper_probability(_frame(None)) is None
    assert extract_whisper_probability(_frame([])) is None
