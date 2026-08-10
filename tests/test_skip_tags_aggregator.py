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

    async def test_flush_unclosed_tag_sentence_mode(self):
        """Unclosed tag: flush still returns the buffered text (pin current behavior).

        Tags are pass-through markers, so an unclosed tag's content is spoken
        the same as a closed tag's, matching test_basic_tags.
        """
        await self.aggregator.reset()

        text = "My email is <spell>foo@pipecat.ai"
        results = [agg async for agg in self.aggregator.aggregate(text)]
        self.assertEqual(len(results), 0)

        result = await self.aggregator.flush()
        self.assertEqual(result.text, text)
        self.assertEqual(result.type, "sentence")

        # Tag state resets after flush, so a reused aggregator isn't stuck
        # thinking it's still inside a tag.
        self.assertEqual(self.aggregator.text.text, "")
        results = [agg async for agg in self.aggregator.aggregate("Hi there. Next")]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Hi there.")


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
        """Flush with an unclosed tag returns the buffered text instead of dropping it."""
        async for _ in self.aggregator.aggregate("<spell>unclosed"):
            pass
        result = await self.aggregator.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "<spell>unclosed")
        self.assertEqual(result.type, "token")

        # Buffer and tag state are reset after flush.
        self.assertEqual(self.aggregator.text.text, "")
        results = [agg async for agg in self.aggregator.aggregate("more text")]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "more text")

    async def test_token_flush_empty_buffer(self):
        """Flush with nothing buffered returns None."""
        async for _ in self.aggregator.aggregate("Hello!"):
            pass
        result = await self.aggregator.flush()
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

    async def test_token_start_tag_split_across_chunks(self):
        """A start tag split across chunks is not leaked as plain text.

        Regression test for https://github.com/pipecat-ai/pipecat/issues/5267:
        on unmodified main, "<spe" is yielded as a token before the tag
        reassembles, fragmenting content that should be emitted as one token.
        """
        results = []
        for token in ["Call ", "<spe", "ll>a b c</spell>", " now"]:
            async for agg in self.aggregator.aggregate(token):
                results.append(agg)

        texts = [r.text for r in results]
        self.assertEqual(texts, ["Call ", "<spell>a b c</spell>", " now"])

    async def test_token_start_tag_split_across_three_chunks(self):
        """A start tag split across three chunks reassembles into one token."""
        results = []
        for token in ["<sp", "el", "l>abc</spell>"]:
            async for agg in self.aggregator.aggregate(token):
                results.append(agg)

        # Nothing should have leaked while the tag was reassembling.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "<spell>abc</spell>")

    async def test_token_partial_tag_chunk_yields_nothing(self):
        """A chunk that is entirely a partial start tag yields nothing."""
        results = [agg async for agg in self.aggregator.aggregate("<spe")]
        self.assertEqual(results, [])
        self.assertEqual(self.aggregator.text.text, "<spe")

    async def test_token_trailing_prefix_char_held_then_emitted(self):
        """A plain trailing '<' is held back and emitted with the next chunk once it no longer matches a tag."""
        results = []
        async for agg in self.aggregator.aggregate("Hi <"):
            results.append(agg)
        # "<" could be the start of "<spell>", so it's held back.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Hi ")

        async for agg in self.aggregator.aggregate(" there"):
            results.append(agg)
        # Once the held-back "<" is followed by something other than "s", it
        # no longer matches a tag prefix and is emitted with the rest.
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1].text, "< there")

    async def test_token_partial_tag_after_yield_in_same_chunk(self):
        """A chunk holding both yieldable text and a trailing partial tag keeps
        the tag working: the retained partial tag is rescanned once complete,
        so content split across later chunks still buffers as one unit.
        """
        results = []
        for token in ["Call <spe", "ll>a b", " c</sp", "ell> now"]:
            async for agg in self.aggregator.aggregate(token):
                results.append(agg)

        texts = [r.text for r in results]
        self.assertEqual(texts, ["Call ", "<spell>a b c</spell> now"])

    async def test_token_multiple_tag_pairs_split_across_chunks(self):
        """With multiple registered tag pairs, a split start tag of either
        pair is held back and reassembles.
        """
        from pipecat.utils.text.base_text_aggregator import AggregationType

        aggregator = SkipTagsAggregator(
            [("<spell>", "</spell>"), ("<code>", "</code>")],
            aggregation_type=AggregationType.TOKEN,
        )

        results = []
        for token in ["Call ", "<cod", "e>x</code>", " and ", "<spe", "ll>y</spell>", " done"]:
            async for agg in aggregator.aggregate(token):
                results.append(agg)

        texts = [r.text for r in results]
        self.assertEqual(texts, ["Call ", "<code>x</code>", " and ", "<spell>y</spell>", " done"])

    async def test_token_second_tag_after_closed_tag_not_fragmented(self):
        """A second tag opened after an earlier tag closes is not fragmented.

        Regression test for https://github.com/pipecat-ai/pipecat/issues/5267:
        on unmodified main, `_current_tag_index` is left pointing past the end
        of the buffer once it's cleared after the first tag closes, making
        `parse_start_end_tags` blind to every tag that follows.
        """
        results = []
        for token in ["<spell>abc</spell>", "<spell>de", "f</spell>", " done"]:
            async for agg in self.aggregator.aggregate(token):
                results.append(agg)

        texts = [r.text for r in results]
        self.assertEqual(texts, ["<spell>abc</spell>", "<spell>def</spell>", " done"])


if __name__ == "__main__":
    unittest.main()
