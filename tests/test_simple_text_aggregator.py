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
        assert self.aggregator.text.text == "Hello"
        await self.aggregator.reset()
        assert self.aggregator.text.text == ""

    async def test_simple_sentence(self):
        assert await self.aggregator.aggregate("Hello ") == None
        aggregate = await self.aggregator.aggregate("Pipecat!")
        assert aggregate.text == "Hello Pipecat!"
        assert aggregate.type == "sentence"
        assert self.aggregator.text.text == ""

    async def test_multiple_sentences(self):
        aggregate = await self.aggregator.aggregate("Hello Pipecat! How are ")
        assert aggregate.text == "Hello Pipecat!"
        # Aggregators should strip leading/trailing spaces when returning text
        assert self.aggregator.text.text == "How are"
        aggregate = await self.aggregator.aggregate("you?")
        assert aggregate.text == "How are you?"

    async def test_word_by_word(self):
        """Test word-by-word token aggregation (e.g., OpenAI)."""
        assert await self.aggregator.aggregate("Hello") == None
        aggregate = await self.aggregator.aggregate("!")
        assert aggregate.text == "Hello!"
        assert await self.aggregator.aggregate(" I") == None
        assert await self.aggregator.aggregate(" am") == None
        aggregate = await self.aggregator.aggregate(" Doug.")
        assert aggregate.text == "I am Doug."
        assert self.aggregator.text.text == ""

    async def test_chunks_with_partial_sentences(self):
        """Test chunks with partial sentences."""
        aggregate = await self.aggregator.aggregate("Hey!")
        assert aggregate.text == "Hey!"
        aggregate = await self.aggregator.aggregate(" Nice to meet you! So")
        assert aggregate.text == "Nice to meet you!"
        assert self.aggregator.text.text == "So"
        assert await self.aggregator.aggregate(" what") == None
        aggregate = await self.aggregator.aggregate("'d you like?")
        assert aggregate.text == "So what'd you like?"

    async def test_multi_sentence_chunk(self):
        """Test chunks with multiple complete sentences."""
        aggregate = await self.aggregator.aggregate("Hello! I am Doug. Nice to meet you!")
        assert aggregate.text == "Hello!"
        # Drain remaining sentences by calling aggregate("")
        aggregate = await self.aggregator.aggregate("")
        assert aggregate.text == "I am Doug."
        aggregate = await self.aggregator.aggregate("")
        assert aggregate.text == "Nice to meet you!"
        assert await self.aggregator.aggregate("") == None
        assert self.aggregator.text.text == ""

    async def test_aggregate_empty_with_incomplete(self):
        """Test aggregate('') with incomplete sentence in buffer."""
        aggregate = await self.aggregator.aggregate("Hello! I am")
        assert aggregate.text == "Hello!"
        assert await self.aggregator.aggregate("") == None
        assert self.aggregator.text.text == "I am"

    async def test_aggregate_empty_buffer(self):
        """Test aggregate('') with empty buffer."""
        assert await self.aggregator.aggregate("") == None
