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
        result = await self.aggregator.aggregate("Hello Pipecat!")
        self.assertEqual(result, "Hello Pipecat!")
        self.assertEqual(self.aggregator.text, "")

    async def test_basic_tags(self):
        await self.aggregator.reset()

        # Tags involved, avoid aggregation during tags.
        result = await self.aggregator.aggregate("My email is <spell>foo@pipecat.ai</spell>.")
        self.assertEqual(result, "My email is <spell>foo@pipecat.ai</spell>.")
        self.assertEqual(self.aggregator.text, "")

    async def test_streaming_tags(self):
        await self.aggregator.reset()

        # Tags involved, stream small chunk of texts.
        result = await self.aggregator.aggregate("My email is <sp")
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, "My email is <sp")

        result = await self.aggregator.aggregate("ell>foo.")
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, "My email is <spell>foo.")

        result = await self.aggregator.aggregate("bar@pipecat.")
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, "My email is <spell>foo.bar@pipecat.")

        result = await self.aggregator.aggregate("ai</spe")
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, "My email is <spell>foo.bar@pipecat.ai</spe")

        result = await self.aggregator.aggregate("ll>.")
        self.assertEqual(result, "My email is <spell>foo.bar@pipecat.ai</spell>.")
        self.assertEqual(self.aggregator.text, "")
