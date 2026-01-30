#
# Copyright (c) 2024-2026, Daily
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
        text = "Hello Pipecat!"
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # Should still be waiting for lookahead after "!"
        self.assertEqual(len(results), 0)

        # Flush to get the pending sentence
        result = await self.aggregator.flush()
        self.assertEqual(result.text, "Hello Pipecat!")
        self.assertEqual(result.type, "sentence")
        self.assertEqual(self.aggregator.text.text, "")

    async def test_basic_tags(self):
        await self.aggregator.reset()

        # Tags involved, avoid aggregation during tags.
        text = "My email is <spell>foo@pipecat.ai</spell>."
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # Should still be waiting for lookahead after "."
        self.assertEqual(len(results), 0)

        # Flush to get the pending sentence
        result = await self.aggregator.flush()
        self.assertEqual(result.text, "My email is <spell>foo@pipecat.ai</spell>.")
        self.assertEqual(result.type, "sentence")
        self.assertEqual(self.aggregator.text.text, "")

    async def test_streaming_tags(self):
        await self.aggregator.reset()

        # Tags involved
        text = "My email is <spell>foo.bar@pipecat.ai</spell>."
        results = [agg async for agg in self.aggregator.aggregate(text)]

        # Should still be waiting for lookahead after "."
        self.assertEqual(len(results), 0)
        self.assertEqual(self.aggregator.text.text, text)
        self.assertEqual(self.aggregator.text.type, "sentence")

        # Flush to get the pending sentence
        result = await self.aggregator.flush()
        self.assertEqual(result.text, text)
        self.assertEqual(self.aggregator.text.text, "")
        self.assertEqual(self.aggregator.text.type, "sentence")


if __name__ == "__main__":
    unittest.main()
