#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for bounded text-transform lookback in token-mode TTS."""

from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
)
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.transforms.replacements import replace_text

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


class _RecordingTTSService(TTSService):
    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=True,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self.requests: list[str] = []

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.requests.append(text)
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


def _response_frames(*text: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(part) for part in text),
        LLMFullResponseEndFrame(),
    ]


def test_replace_text_rejects_negative_lookback():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        replace_text([("x", "y")], lookback_max_chars=-1)


@pytest.mark.asyncio
async def test_token_transform_matches_across_token_boundary():
    transform = replace_text([(r"50\+", "fifty plus")], lookback_max_chars=3)
    tts = _RecordingTTSService(
        text_aggregation_mode=TextAggregationMode.TOKEN,
        text_transforms=[("*", transform)],
    )

    await run_test(tts, frames_to_send=_response_frames("Pay ", "50", "+", " now"))

    assert "".join(tts.requests) == "Pay fifty plus now"


@pytest.mark.asyncio
async def test_token_transform_releases_text_outside_lookback_immediately():
    transform = replace_text([("xyz", "abc")], lookback_max_chars=3)
    tts = _RecordingTTSService(
        text_aggregation_mode=TextAggregationMode.TOKEN,
        text_transforms=[("*", transform)],
    )

    await run_test(tts, frames_to_send=_response_frames("Hello"))

    assert tts.requests == ["He", "llo"]


@pytest.mark.asyncio
async def test_token_transform_flushes_held_match_once_at_response_end():
    transform = replace_text([(r"50\+", "fifty plus")], lookback_max_chars=3)
    tts = _RecordingTTSService(
        text_aggregation_mode=TextAggregationMode.TOKEN,
        text_transforms=[("*", transform)],
    )

    await run_test(tts, frames_to_send=_response_frames("50", "+"))

    assert tts.requests == ["fifty plus"]


@pytest.mark.asyncio
async def test_token_transform_discards_held_text_on_interruption():
    transform = replace_text([(r"50\+", "fifty plus")], lookback_max_chars=3)
    tts = _RecordingTTSService(
        text_aggregation_mode=TextAggregationMode.TOKEN,
        text_transforms=[("*", transform)],
    )
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("50"),
        SleepFrame(sleep=0.05),
        InterruptionFrame(),
        LLMTextFrame("+"),
        LLMFullResponseEndFrame(),
    ]

    await run_test(tts, frames_to_send=frames)

    assert tts.requests == ["+"]


@pytest.mark.asyncio
async def test_replace_text_without_lookback_keeps_token_behavior():
    transform = replace_text([(r"50\+", "fifty plus")])
    tts = _RecordingTTSService(
        text_aggregation_mode=TextAggregationMode.TOKEN,
        text_transforms=[("*", transform)],
    )

    await run_test(tts, frames_to_send=_response_frames("50", "+"))

    assert tts.requests == ["50", "+"]


@pytest.mark.asyncio
async def test_replace_text_lookback_does_not_change_sentence_mode():
    transform = replace_text([(r"50\+", "fifty plus")], lookback_max_chars=3)
    tts = _RecordingTTSService(
        text_aggregation_mode=TextAggregationMode.SENTENCE,
        text_transforms=[("*", transform)],
    )

    await run_test(tts, frames_to_send=_response_frames("Pay 50", "+ now."))

    assert tts.requests == ["Pay fifty plus now."]
