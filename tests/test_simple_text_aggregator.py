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
