#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the transcription probability helpers in ``pipecat.services.whisper.utils``."""

import math

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.whisper.utils import extract_openai_gpt4o_probability

Transcription = pytest.importorskip("openai.types.audio").Transcription


def _frame(result) -> TranscriptionFrame:
    return TranscriptionFrame(text="hello there", user_id="user", timestamp="", result=result)


def test_gpt4o_probability_averages_token_logprobs():
    response = Transcription.model_validate(
        {
            "text": "hello there",
            "logprobs": [
                {"token": "hello", "logprob": -0.1, "bytes": [104, 101, 108, 108, 111]},
                {"token": " there", "logprob": -0.3, "bytes": [32, 116, 104, 101, 114, 101]},
            ],
        }
    )

    probability = extract_openai_gpt4o_probability(_frame(response))

    assert probability == pytest.approx(math.exp(-0.2))


def test_gpt4o_probability_is_none_without_logprobs():
    response = Transcription.model_validate({"text": "hello there"})

    assert extract_openai_gpt4o_probability(_frame(response)) is None
