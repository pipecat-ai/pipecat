#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Numbered list markers stay attached to their streamed sentence."""

import pytest

from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 3, 1000])
@pytest.mark.parametrize(
    "sentences",
    [
        ["1. The morning sun warmed the quiet street.", "\n2. Birds sang."],
        ["12. First item.", "\n  13. Next item."],
        ["Options:\n1. Take a walk.", "\n2. Read a book."],
        ["1. 👋 Welcome.", "\n2. Café is open."],
        ["1.25 dollars is the price.", "Next sentence."],
        ["Options follow.", "\n1. Take a walk.", "\n2. Read a book."],
        ["It costs 1.25 dollars.", "The answer is 1.", "Next question."],
    ],
)
async def test_streamed_numbered_lists(chunk_size, sentences):
    text = " ".join(sentences)
    aggregator = SimpleTextAggregator()
    actual = []
    for offset in range(0, len(text), chunk_size):
        actual.extend(
            [a.text async for a in aggregator.aggregate(text[offset : offset + chunk_size])]
        )
    tail = await aggregator.flush()
    if tail:
        actual.append(tail.text)
    assert actual == sentences


@pytest.mark.asyncio
async def test_numbered_item_waits_for_content_and_clears_on_interruption():
    aggregator = SimpleTextAggregator()
    assert [a.text async for a in aggregator.aggregate("1. T")] == []
    await aggregator.handle_interruption()
    assert [a.text async for a in aggregator.aggregate("Hello. Next")] == ["Hello."]
    assert (await aggregator.flush()).text == "Next"


@pytest.mark.asyncio
async def test_bare_number_is_preserved_when_response_finishes():
    aggregator = SimpleTextAggregator()
    assert [a.text async for a in aggregator.aggregate("1.")] == []
    assert (await aggregator.flush()).text == "1."
