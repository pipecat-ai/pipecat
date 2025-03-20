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
        assert self.aggregator.aggregate("Hello ") == None
        assert self.aggregator.text == "Hello "
        self.aggregator.reset()
        assert self.aggregator.text == ""

    async def test_simple_sentence(self):
        assert self.aggregator.aggregate("Hello ") == None
        assert self.aggregator.aggregate("Pipecat!") == "Hello Pipecat!"
        assert self.aggregator.text == ""

    async def test_multiple_sentences(self):
        assert self.aggregator.aggregate("Hello Pipecat! How are ") == "Hello Pipecat!"
        assert self.aggregator.aggregate("you?") == " How are you?"
