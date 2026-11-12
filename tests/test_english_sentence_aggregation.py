#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""English sentence boundaries under fragmented streaming input."""

import re

import pytest

from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


@pytest.mark.asyncio
@pytest.mark.parametrize("chunking", ["whole", "word", "character", "three_characters"])
@pytest.mark.parametrize(
    "sentences",
    [
        pytest.param(["We met at 5 p.m.", "Then we left."], id="sentence_final_time"),
        pytest.param(
            ["He lives in the U.S.", "His sister lives in Canada."], id="sentence_final_acronym"
        ),
        pytest.param(
            ["We sell pens, paper, etc. in the shop.", "Come inside."],
            id="mid_sentence_abbreviation",
        ),
        pytest.param(
            ["We sell pens, paper, etc.", "The shop closes at six."],
            id="sentence_final_abbreviation",
        ),
        pytest.param(
            ["W. E. B. Du Bois wrote extensively.", "His work remains influential."], id="initials"
        ),
        pytest.param(['"Stop!" he shouted.', "The car halted."], id="attributed_exclamation"),
        pytest.param(["“Are you ready?” she asked.", "I nodded."], id="attributed_question"),
        pytest.param(["Wait...", "Then continue."], id="ellipsis_boundary"),
        pytest.param(
            ["That's strange... but possible.", "Let's check."], id="mid_sentence_ellipsis"
        ),
        pytest.param(["No!!!", "Stop!"], id="repeated_exclamation"),
        pytest.param(
            ["Your IP address is 192.168.1.1.", "Enter it in the browser."], id="ip_address"
        ),
        pytest.param(["Go to https://example.com/help.", "Then click Support."], id="url"),
        pytest.param(["Email support@example.com.", "We'll reply soon."], id="email"),
    ],
)
async def test_english_boundaries_preserve_text_across_chunks(chunking, sentences):
    text = " ".join(sentences)
    if chunking == "whole":
        chunks = [text]
    elif chunking == "word":
        chunks = re.findall(r"\S+\s*", text)
    else:
        size = 1 if chunking == "character" else 3
        chunks = [text[i : i + size] for i in range(0, len(text), size)]

    aggregator = SimpleTextAggregator()
    actual = []
    for chunk in chunks:
        actual.extend([item.text async for item in aggregator.aggregate(chunk)])
    tail = await aggregator.flush()
    if tail:
        actual.append(tail.text)

    assert actual == sentences
    assert await aggregator.flush() is None
