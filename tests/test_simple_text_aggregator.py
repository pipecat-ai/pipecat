#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class TestSimpleTextAggregator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.aggregator = SimpleTextAggregator()

    async def test_reset_aggregations(self):
        assert await self.aggregator.aggregate("Hello ") == None
        assert self.aggregator.text == "Hello "
        await self.aggregator.reset()
        assert self.aggregator.text == ""

    async def test_word_by_word(self):
        """Test word-by-word token aggregation (e.g., OpenAI)."""
        assert await self.aggregator.aggregate("Hello") == None
        assert await self.aggregator.aggregate("!") == "Hello!"
        assert await self.aggregator.aggregate(" I") == None
        assert await self.aggregator.aggregate(" am") == None
        assert await self.aggregator.aggregate(" Doug.") == " I am Doug."
        assert self.aggregator.text == ""

    async def test_chunks_with_partial_sentences(self):
        """Test chunks with partial sentences."""
        assert await self.aggregator.aggregate("Hey!") == "Hey!"
        assert await self.aggregator.aggregate(" Nice to meet you! So") == " Nice to meet you!"
        assert self.aggregator.text == " So"
        assert await self.aggregator.aggregate(" what") == None
        assert await self.aggregator.aggregate("'d you like?") == " So what'd you like?"

    async def test_multi_sentence_chunk(self):
        """Test chunks with multiple complete sentences."""
        result = await self.aggregator.aggregate("Hello! I am Doug. Nice to meet you!")
        assert result == "Hello!"
        # Drain remaining sentences
        assert await self.aggregator.flush_next_sentence() == " I am Doug."
        assert await self.aggregator.flush_next_sentence() == " Nice to meet you!"
        assert await self.aggregator.flush_next_sentence() == None
        assert self.aggregator.text == ""

    async def test_flush_next_sentence_with_incomplete(self):
        """Test flush_next_sentence with incomplete sentence in buffer."""
        assert await self.aggregator.aggregate("Hello! I am") == "Hello!"
        assert await self.aggregator.flush_next_sentence() == None
        assert self.aggregator.text == " I am"

    async def test_flush_next_sentence_empty_buffer(self):
        """Test flush_next_sentence with empty buffer."""
        assert await self.aggregator.flush_next_sentence() == None
