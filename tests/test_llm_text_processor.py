#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the text transforms of :class:`LLMTextProcessor`."""

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.tests.utils import run_test
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
