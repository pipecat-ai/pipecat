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


class TestSkipTagsAggregatorTokenMode(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from pipecat.utils.text.base_text_aggregator import AggregationType

        self.aggregator = SkipTagsAggregator(
            [("<spell>", "</spell>")], aggregation_type=AggregationType.TOKEN
        )

    async def test_token_no_tags(self):
        """No tags: text passes through immediately as TOKEN."""
        results = [agg async for agg in self.aggregator.aggregate("Hello!")]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Hello!")
        self.assertEqual(results[0].type, "token")

    async def test_token_inside_tag_buffers(self):
        """Inside a tag, text is buffered until the closing tag is found."""
        results = [agg async for agg in self.aggregator.aggregate("<spell>foo@bar")]
        # Still inside tag, nothing yielded
        self.assertEqual(len(results), 0)

        # Close the tag
        results = [agg async for agg in self.aggregator.aggregate("</spell>")]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "<spell>foo@bar</spell>")
        self.assertEqual(results[0].type, "token")

    async def test_token_flush_unclosed_tag(self):
        """Flush with unclosed tag returns remaining text."""
        async for _ in self.aggregator.aggregate("<spell>unclosed"):
            pass
        result = await self.aggregator.flush()
        # TOKEN mode flush returns None (parent behavior)
        self.assertIsNone(result)

    async def test_token_text_around_tags(self):
        """Simulate word-by-word token delivery with tags."""
        results = []
        # Simulate LLM streaming tokens one at a time
        for token in ["Hi ", "<spell>", "X", "</spell>", " bye"]:
            async for agg in self.aggregator.aggregate(token):
                results.append(agg)

        self.assertEqual(len(results), 3)
        # Text before tag passes through immediately
        self.assertEqual(results[0].text, "Hi ")
        self.assertEqual(results[0].type, "token")
        # Tagged content is buffered until the closing tag, then yielded whole
        self.assertEqual(results[1].text, "<spell>X</spell>")
        self.assertEqual(results[1].type, "token")
        # Text after tag passes through immediately
        self.assertEqual(results[2].text, " bye")
        self.assertEqual(results[2].type, "token")


if __name__ == "__main__":
    unittest.main()
