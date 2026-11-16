#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :class:`LLMTextProcessor`."""

from unittest.mock import AsyncMock

import pytest

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.pattern_pair_aggregator import MatchAction, PatternPairAggregator


async def upper(text: str, text_type: str) -> str:
    return text.upper()


async def drop(text: str, text_type: str) -> str:
    return ""


async def fail(text: str, text_type: str) -> str:
    raise ValueError("boom")


def _response(*texts: str) -> list[Frame]:
    return [LLMFullResponseStartFrame(), *map(LLMTextFrame, texts), LLMFullResponseEndFrame()]


def _aggregated(frames) -> list[AggregatedTextFrame]:
    return [f for f in frames if isinstance(f, AggregatedTextFrame)]


async def _aggregations(processor: LLMTextProcessor, *texts: str):
    down, up = await run_test(processor, frames_to_send=_response(*texts))
    return _aggregated(down), up


@pytest.mark.asyncio
async def test_transform_rewrites_text_and_raw_text():
    """A "*" transform rewrites every aggregation, flushed text included."""
    plain, _ = await _aggregations(LLMTextProcessor(), "Hello there. ", "Bye")
    frames, _ = await _aggregations(
        LLMTextProcessor(text_transforms=[("*", upper)]), "Hello there. ", "Bye"
    )
    assert [f.text for f in frames] == [f.text.upper() for f in plain]
    assert [f.raw_text for f in frames] == [f.text.upper() for f in plain]
    assert [f.text_type for f in frames] == [f.text_type for f in plain]


def _code_aggregator() -> PatternPairAggregator:
    aggregator = PatternPairAggregator()
    aggregator.add_pattern("code", "<code>", "</code>", MatchAction.AGGREGATE)
    return aggregator


@pytest.mark.asyncio
async def test_transform_applies_to_its_type_only():
    """A transform for one type leaves the other types alone and keeps delimiters."""
    text = "Run this <code>print(1)</code> now. "
    plain, _ = await _aggregations(LLMTextProcessor(text_aggregator=_code_aggregator()), text)
    frames, _ = await _aggregations(
        LLMTextProcessor(text_aggregator=_code_aggregator(), text_transforms=[("code", upper)]),
        text,
    )

    code = [f for f in frames if f.text_type == "code"]
    assert len(code) == 1
    assert code[0].text == "PRINT(1)"
    assert code[0].raw_text == "<code>PRINT(1)</code>"
    others = [(f.text, f.raw_text) for f in frames if f.text_type != "code"]
    assert others == [(f.text, f.raw_text) for f in plain if f.text_type != "code"]
    assert len(others) == 2


@pytest.mark.asyncio
async def test_transform_leaving_whitespace_drops_the_aggregation():
    frames, _ = await _aggregations(
        LLMTextProcessor(text_transforms=[("*", drop)]), "Leaked reasoning. "
    )
    assert frames == []


@pytest.mark.asyncio
async def test_failing_transform_drops_the_aggregation_and_reports():
    frames, up = await _aggregations(
        LLMTextProcessor(text_transforms=[("*", fail)]), "Hello there. "
    )
    assert frames == []
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert "boom" in errors[0].error


@pytest.mark.asyncio
async def test_add_and_remove_text_transformer():
    processor = LLMTextProcessor()
    processor.add_text_transformer(upper)
    processor.remove_text_transformer(upper)
    frames, _ = await _aggregations(processor, "Hello there. ")
    assert [f.text for f in frames] == ["Hello there."]


class _OutputTransport(BaseOutputTransport):
    """An output transport that takes every audio frame without playing it."""

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        return True


def _speech(context_id: str) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(
        audio=b"\x10\x27" * 1600, sample_rate=16000, num_channels=1, context_id=context_id
    )


@pytest.mark.asyncio
async def test_held_text_is_pushed_before_tts_stopped():
    """A speech-to-speech service stops its speech before it ends the response.

    The last sentence has no following text to close it, so the processor
    still holds it when the speech stops. It was spoken before the stop and
    goes out ahead of it.
    """
    down_frames, _ = await run_test(
        LLMTextProcessor(),
        frames_to_send=[
            LLMTextFrame("Hello there. "),
            LLMTextFrame("It's provider-agnostic."),
            TTSStoppedFrame(),
            LLMFullResponseEndFrame(),
        ],
        expected_down_frames=[
            AggregatedTextFrame,
            AggregatedTextFrame,
            TTSStoppedFrame,
            LLMFullResponseEndFrame,
        ],
    )
    assert [f.text for f in _aggregated(down_frames)] == [
        "Hello there.",
        "It's provider-agnostic.",
    ]


@pytest.mark.asyncio
async def test_speech_to_speech_reply_ends_with_its_last_sentence():
    """A reply's last sentence reaches clients before the user speaks again.

    Text that reaches the output transport after the bot stopped speaking
    waits for the next speech, and the user's next turn interrupts before
    that.
    """
    observer = RTVIObserver(params=RTVIObserverParams())
    observer.send_rtvi_message = AsyncMock()
    transport = _OutputTransport(TransportParams(audio_out_enabled=True))

    await run_test(
        Pipeline([LLMTextProcessor(), transport]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            TTSStartedFrame(context_id="reply"),
            _speech("reply"),
            LLMTextFrame("Hello there. "),
            _speech("reply"),
            LLMTextFrame("It's provider-agnostic."),
            TTSStoppedFrame(context_id="reply"),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
            # The user starts their next turn.
            InterruptionFrame(),
            SleepFrame(sleep=0.1),
        ],
        observers=[observer],
    )

    sent = [
        call.args[0].data.text
        for call in observer.send_rtvi_message.await_args_list
        if isinstance(call.args[0], RTVI.BotOutputMessage)
    ]
    assert sent == ["Hello there.", "It's provider-agnostic."]
