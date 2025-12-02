#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator


class TestSkipTagsAggregator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.aggregator = SkipTagsAggregator([("<spell>", "</spell>")])

    async def test_no_tags(self):
        await self.aggregator.reset()

        # No tags involved, aggregate at end of sentence.
        # Feed text character by character
        result = None
        for char in "Hello Pipecat!":
            result = await self.aggregator.aggregate(char)
            if result:
                break

        # Should still be waiting for lookahead after "!"
        self.assertIsNone(result)

        # Flush to get the pending sentence
        result = await self.aggregator.flush()
        self.assertEqual(result.text, "Hello Pipecat!")
        self.assertEqual(result.type, "sentence")
        self.assertEqual(self.aggregator.text.text, "")

    async def test_basic_tags(self):
        await self.aggregator.reset()

        # Tags involved, avoid aggregation during tags.
        # Feed text character by character
        result = None
        for char in "My email is <spell>foo@pipecat.ai</spell>.":
            result = await self.aggregator.aggregate(char)
            if result:
                break

        # Should still be waiting for lookahead after "."
        self.assertIsNone(result)

        # Flush to get the pending sentence
        result = await self.aggregator.flush()
        self.assertEqual(result.text, "My email is <spell>foo@pipecat.ai</spell>.")
        self.assertEqual(result.type, "sentence")
        self.assertEqual(self.aggregator.text.text, "")

    async def test_streaming_tags(self):
        await self.aggregator.reset()

        # Tags involved, feed character by character
        full_text = "My email is <spell>foo.bar@pipecat.ai</spell>."
        result = None

        for char in full_text:
            result = await self.aggregator.aggregate(char)
            if result:
                break

        # Should still be waiting for lookahead after "."
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text.text, full_text)
        self.assertEqual(self.aggregator.text.type, "sentence")

        # Flush to get the pending sentence
        result = await self.aggregator.flush()
        self.assertEqual(result.text, full_text)
        self.assertEqual(self.aggregator.text.text, "")
        self.assertEqual(self.aggregator.text.type, "sentence")
