#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class TestSimpleTextAggregator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.aggregator = SimpleTextAggregator()

    async def test_reset_aggregations(self):
        text = "Hello "
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # No complete sentences yet
        assert len(results) == 0
        assert self.aggregator.text.text == "Hello"
        await self.aggregator.reset()
        assert self.aggregator.text.text == ""

    async def test_simple_sentence(self):
        text = "Hello Pipecat!"
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # No complete sentences yet (waiting for lookahead after "!")
        assert len(results) == 0

        # Flush to get the pending sentence
        aggregate = await self.aggregator.flush()
        assert aggregate.text == "Hello Pipecat!"
        assert aggregate.type == "sentence"
        assert self.aggregator.text.text == ""

    async def test_multiple_sentences(self):
        text = "Hello Pipecat! How are you?"
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # First sentence should be complete (lookahead from "H" confirmed it)
        assert len(results) == 1
        assert results[0].text == "Hello Pipecat!"

        # Flush to get the pending sentence
        result = await self.aggregator.flush()
        assert result.text == "How are you?"

    async def test_lookahead_decimal_number(self):
        """Test that $29.95 is not split at $29."""
        text = "Ask me for only $29.95/month."
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # No complete sentences yet (waiting for lookahead after final ".")
        assert len(results) == 0

        # Can use flush() to get the pending sentence at end of stream
        result = await self.aggregator.flush()
        assert result.text == "Ask me for only $29.95/month."

    async def test_lookahead_abbreviation(self):
        """Test that Mr. Smith is not split at Mr."""
        text = "Hello Mr. Smith."
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # No complete sentences yet (waiting for lookahead after final ".")
        assert len(results) == 0

        # Can use flush() to get the pending sentence at end of stream
        result = await self.aggregator.flush()
        assert result.text == "Hello Mr. Smith."

    async def test_lookahead_actual_sentence_end(self):
        """Test that a real sentence end is detected after lookahead."""
        text = "Hello world. Next sentence"
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # First sentence should be complete (lookahead from "N" confirmed it)
        assert len(results) == 1
        assert results[0].text == "Hello world."

    async def test_flush_pending_sentence(self):
        """Test that flush() returns pending sentence waiting for lookahead."""
        text = "Hello world."
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # No complete sentences yet (waiting for lookahead)
        assert len(results) == 0

        # Call flush to get it
        result = await self.aggregator.flush()
        assert result is not None
        assert result.text == "Hello world."
        # Flush again should return None
        assert await self.aggregator.flush() == None

    async def test_flush_with_no_pending(self):
        """Test that flush() returns any remaining text in buffer."""
        text = "Hello"
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # No complete sentences
        assert len(results) == 0

        result = await self.aggregator.flush()
        # flush() now returns any remaining text, not just pending lookahead
        assert result is not None
        assert result.text == "Hello"
        # Buffer should be empty after flush
        assert self.aggregator.text.text == ""

    async def test_flush_after_lookahead_confirmed(self):
        """Test flush after lookahead has already confirmed sentence."""
        text = "Hello. W"
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # First sentence should be complete (lookahead from "W" confirmed it)
        assert len(results) == 1
        assert results[0].text == "Hello."

        # flush() returns any remaining text (the "W" in this case)
        result = await self.aggregator.flush()
        assert result.text == "W"
