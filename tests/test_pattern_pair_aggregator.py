#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock

from pipecat.utils.text.pattern_pair_aggregator import (
    MatchAction,
    PatternMatch,
    PatternPairAggregator,
)


class TestPatternPairAggregator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.aggregator = PatternPairAggregator()
        self.test_handler = AsyncMock()
        self.code_handler = AsyncMock()

        # Add a test pattern
        self.aggregator.add_pattern(
            type="test_pattern",
            start_pattern="<test>",
            end_pattern="</test>",
        )
        self.aggregator.add_pattern(
            type="code_pattern",
            start_pattern="<code>",
            end_pattern="</code>",
            action=MatchAction.AGGREGATE,
        )

        # Register the mock handler
        self.aggregator.on_pattern_match("test_pattern", self.test_handler)
        self.aggregator.on_pattern_match("code_pattern", self.code_handler)

    async def test_pattern_match_and_removal(self):
        text = "Hello <test>pattern content</test>!"
        results = [result async for result in self.aggregator.aggregate(text)]

        # Verify the handler was called with correct PatternMatch object
        self.test_handler.assert_called_once()
        call_args = self.test_handler.call_args[0][0]
        self.assertIsInstance(call_args, PatternMatch)
        self.assertEqual(call_args.type, "test_pattern")
        self.assertEqual(call_args.full_match, "<test>pattern content</test>")
        self.assertEqual(call_args.text, "pattern content")

        # No results yet (waiting for lookahead after "!")
        self.assertEqual(len(results), 0)

        # Next sentence should provide the lookahead and trigger the previous sentence
        async for result in self.aggregator.aggregate(" This is another sentence."):
            results.append(result)

        # First result should be "Hello !" triggered by the space lookahead
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Hello !")
        self.assertEqual(results[0].type, "sentence")

        # Now flush to get the remaining sentence
        result = await self.aggregator.flush()
        self.assertEqual(result.text, "This is another sentence.")

        # Buffer should be empty after returning a complete sentence
        self.assertEqual(self.aggregator.text.text, "")

    async def test_pattern_match_and_aggregate(self):
        text = "Here is code <code>pattern content</code> This is another sentence."

        results = [result async for result in self.aggregator.aggregate(text)]

        # First result should be "Here is code" when pattern starts
        self.assertEqual(results[0].text, "Here is code")
        self.assertEqual(results[0].type, "sentence")

        # Second result should be the code pattern content
        self.assertEqual(results[1].text, "pattern content")
        self.assertEqual(results[1].type, "code_pattern")

        # Verify the handler was called with correct PatternMatch object
        self.code_handler.assert_called_once()
        call_args = self.code_handler.call_args[0][0]
        self.assertIsInstance(call_args, PatternMatch)
        self.assertEqual(call_args.type, "code_pattern")
        self.assertEqual(call_args.full_match, "<code>pattern content</code>")
        self.assertEqual(call_args.text, "pattern content")

        # Last sentence needs flush (waiting for lookahead after ".")
        result = await self.aggregator.flush()
        self.assertEqual(result.text, "This is another sentence.")
        self.assertEqual(result.type, "sentence")

        # Buffer should be empty after returning a complete sentence
        self.assertEqual(self.aggregator.text.text, "")

    async def test_incomplete_pattern(self):
        text = "Hello <test>pattern content"
        results = [result async for result in self.aggregator.aggregate(text)]
        # No complete pattern yet, so nothing should be returned
        self.assertEqual(len(results), 0)

        # The handler should not be called yet
        self.test_handler.assert_not_called()

        # Buffer should contain the incomplete text
        self.assertEqual(self.aggregator.text.text, "Hello <test>pattern content")
        self.assertEqual(self.aggregator.text.type, "test_pattern")

        # Reset and confirm buffer is cleared
        await self.aggregator.reset()
        self.assertEqual(self.aggregator.text.text, "")

    async def test_multiple_patterns(self):
        # Set up multiple patterns and handlers
        voice_handler = AsyncMock()
        emphasis_handler = AsyncMock()

        self.aggregator.add_pattern(
            type="voice",
            start_pattern="<voice>",
            end_pattern="</voice>",
            action=MatchAction.REMOVE,
        )

        self.aggregator.add_pattern(
            type="emphasis",
            start_pattern="<em>",
            end_pattern="</em>",
            action=MatchAction.KEEP,  # Keep emphasis tags
        )

        self.aggregator.on_pattern_match("voice", voice_handler)
        self.aggregator.on_pattern_match("emphasis", emphasis_handler)

        text = "Hello <voice>female</voice> I am <em>very</em> excited to meet you!"
        results = [result async for result in self.aggregator.aggregate(text)]

        # Both handlers should be called with correct data
        voice_handler.assert_called_once()
        voice_match = voice_handler.call_args[0][0]
        self.assertEqual(voice_match.type, "voice")
        self.assertEqual(voice_match.text, "female")

        emphasis_handler.assert_called_once()
        emphasis_match = emphasis_handler.call_args[0][0]
        self.assertEqual(emphasis_match.type, "emphasis")
        self.assertEqual(emphasis_match.text, "very")

        # With lookahead, we need to flush to get the final sentence
        self.assertEqual(len(results), 0)  # Waiting for lookahead after "!"

        result = await self.aggregator.flush()
        # Voice pattern should be removed, emphasis pattern should remain
        self.assertEqual(result.text, "Hello  I am <em>very</em> excited to meet you!")

        # Buffer should be empty
        self.assertEqual(self.aggregator.text.text, "")

    async def test_handle_interruption(self):
        text = "Hello <test>pattern"
        results = [result async for result in self.aggregator.aggregate(text)]
        self.assertEqual(len(results), 0)

        # Simulate interruption
        await self.aggregator.handle_interruption()

        # Buffer should be cleared
        self.assertEqual(self.aggregator.text.text, "")

        # Handler should not have been called
        self.test_handler.assert_not_called()

    async def test_pattern_across_sentences(self):
        text = "Hello <test>This is sentence one. This is sentence two.</test> Final sentence."
        results = [result async for result in self.aggregator.aggregate(text)]

        # Handler should be called with entire content
        self.test_handler.assert_called_once()
        call_args = self.test_handler.call_args[0][0]
        self.assertEqual(call_args.text, "This is sentence one. This is sentence two.")

        # With lookahead, we need to flush to get the final sentence
        self.assertEqual(len(results), 0)  # Waiting for lookahead after "."

        result = await self.aggregator.flush()
        # Pattern should be removed, resulting in text with sentences merged
        self.assertEqual(result.text, "Hello  Final sentence.")

        # Buffer should be empty
        self.assertEqual(self.aggregator.text.text, "")

    async def test_flush_unclosed_pattern_returns_preceding_text(self):
        """Unclosed REMOVE pattern: flush returns text before it, drops the rest.

        A closed <test>...</test> pair is stripped from the output entirely
        (see test_pattern_match_and_removal); an unclosed one degrades to the
        same result instead of leaking the raw start tag and its content.
        """
        text = "Well <test>pattern content"
        results = [result async for result in self.aggregator.aggregate(text)]
        self.assertEqual(len(results), 0)

        result = await self.aggregator.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Well")
        self.assertNotIn("<test>", result.text)
        self.assertNotIn("pattern content", result.text)

        # The pair never closed, so its handler is not invoked.
        self.test_handler.assert_not_called()

        # Buffer is cleared after flush.
        self.assertEqual(self.aggregator.text.text, "")

    async def test_flush_unclosed_pattern_with_no_preceding_text(self):
        """Unclosed pattern with nothing before it: flush drops it entirely."""
        text = "<test>pattern content"
        results = [result async for result in self.aggregator.aggregate(text)]
        self.assertEqual(len(results), 0)

        result = await self.aggregator.flush()
        self.assertIsNone(result)
        self.test_handler.assert_not_called()

    async def test_flush_state_resets_for_reuse(self):
        """After flush drops an incomplete pattern, the aggregator works cleanly again."""
        text = "Well <test>pattern content"
        async for _ in self.aggregator.aggregate(text):
            pass
        await self.aggregator.flush()
        self.assertEqual(self.aggregator.text.text, "")

        # A fresh, fully-closed pattern still works after the reset.
        text = "New <test>value</test> sentence."
        results = [result async for result in self.aggregator.aggregate(text)]
        result = await self.aggregator.flush()
        combined = "".join(r.text for r in results) + (result.text if result else "")
        self.assertNotIn("value", combined)
        self.assertIn("New", combined)

    async def test_flush_remove_pattern_closed_then_unclosed_same_type(self):
        """A prior closed REMOVE pair doesn't confuse handling of a later,
        genuinely unclosed occurrence of the same pattern type: flush must
        find the actual unmatched occurrence, not just the first occurrence
        of the start delimiter in the buffer.
        """
        text = "Start <test>closed</test> middle <test>unclosed"
        results = [result async for result in self.aggregator.aggregate(text)]

        result = await self.aggregator.flush()
        combined = "".join(r.text for r in results) + (result.text if result else "")
        self.assertIn("Start", combined)
        self.assertIn("middle", combined)
        self.assertNotIn("closed", combined)
        self.assertNotIn("unclosed", combined)

        self.test_handler.assert_called_once()

    async def test_flush_earliest_unmatched_wins_regardless_of_registration_order(self):
        """Two different unclosed REMOVE patterns: flush cuts at whichever
        starts first in the text, not whichever pattern was registered first.
        """
        aggregator = PatternPairAggregator()
        voice_handler = AsyncMock()
        test_handler = AsyncMock()
        # Registered in reverse of where they appear in the text below.
        aggregator.add_pattern(
            type="voice", start_pattern="<voice>", end_pattern="</voice>", action=MatchAction.REMOVE
        )
        aggregator.add_pattern(
            type="test2", start_pattern="<test>", end_pattern="</test>", action=MatchAction.REMOVE
        )
        aggregator.on_pattern_match("voice", voice_handler)
        aggregator.on_pattern_match("test2", test_handler)

        text = "Hi <test>foo <voice>bar"
        results = [result async for result in aggregator.aggregate(text)]
        self.assertEqual(len(results), 0)

        result = await aggregator.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Hi")
        self.assertNotIn("foo", result.text)
        self.assertNotIn("bar", result.text)
        voice_handler.assert_not_called()
        test_handler.assert_not_called()

    async def test_flush_unclosed_keep_pattern_kept_verbatim(self):
        """Unclosed KEEP pattern: flush keeps the content, delimiter
        included, just as a closed KEEP pair is kept verbatim (see
        test_multiple_patterns).
        """
        self.aggregator.add_pattern(
            type="emphasis",
            start_pattern="<em>",
            end_pattern="</em>",
            action=MatchAction.KEEP,
        )

        text = "Well <em>unclosed content"
        results = [result async for result in self.aggregator.aggregate(text)]
        self.assertEqual(len(results), 0)

        result = await self.aggregator.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result.text, text)

    async def test_flush_keep_pattern_uses_correct_unmatched_occurrence(self):
        """A closed KEEP pair earlier in the buffer must not be mistaken for
        the unmatched one; the actually-unmatched occurrence can come later.
        """
        self.aggregator.add_pattern(
            type="emphasis",
            start_pattern="<em>",
            end_pattern="</em>",
            action=MatchAction.KEEP,
        )

        text = "Hello <em>bold</em> world <em>unclosed"
        results = [result async for result in self.aggregator.aggregate(text)]
        self.assertEqual(len(results), 0)

        # Unclosed KEEP content is kept, so nothing gets dropped, and the
        # closed <em>bold</em> pair is not mistaken for the unmatched one.
        result = await self.aggregator.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result.text, text)

    async def test_flush_unclosed_aggregate_pattern_dropped_without_handler(self):
        """Unclosed AGGREGATE pattern: content is dropped like REMOVE, and its
        handler is not invoked, since AGGREGATE content is a side channel
        that's never spoken.
        """
        text = "Before <code>unclosed content"
        results = [result async for result in self.aggregator.aggregate(text)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Before")

        result = await self.aggregator.flush()
        self.assertIsNone(result)
        self.code_handler.assert_not_called()

    async def test_flush_skips_completed_pair_still_in_buffer(self):
        """A completed pair still sitting in the buffer at flush time is not
        mistaken for an unmatched one: its start delimiter (and any start
        nested inside its span) is skipped, and the cut happens at the
        genuinely unmatched occurrence that follows it.
        """
        # Seed the buffer directly: aggregate() strips closed REMOVE pairs
        # eagerly, so flush() seeing one only happens in edge cases (e.g. a
        # pair completing while _last_processed_position is stale).
        self.aggregator._text = "Start <test>closed</test> middle <test>unclosed"

        result = await self.aggregator.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Start <test>closed</test> middle")
        self.assertNotIn("unclosed", result.text)

    async def test_flush_trims_trailing_partial_start_delimiter(self):
        """Buffer ending mid-delimiter (stream cut off inside '<test>') has
        the partial delimiter trimmed rather than spoken as plain text.
        """
        text = "Hello <te"
        results = [result async for result in self.aggregator.aggregate(text)]
        self.assertEqual(len(results), 0)

        result = await self.aggregator.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Hello")


class TestPatternPairAggregatorTokenMode(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from pipecat.utils.text.base_text_aggregator import AggregationType

        self.aggregator = PatternPairAggregator(aggregation_type=AggregationType.TOKEN)
        self.handler = AsyncMock()
        self.aggregator.add_pattern(
            type="think",
            start_pattern="<think>",
            end_pattern="</think>",
            action=MatchAction.REMOVE,
        )
        self.aggregator.on_pattern_match("think", self.handler)

    async def test_token_no_patterns(self):
        """Non-pattern text passes through as TOKEN, one per aggregate call."""
        results = []
        for token in ["Hello", " world", "."]:
            async for r in self.aggregator.aggregate(token):
                results.append(r)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].text, "Hello")
        self.assertEqual(results[1].text, " world")
        self.assertEqual(results[2].text, ".")
        for r in results:
            self.assertEqual(r.type, "token")

    async def test_token_pattern_detection(self):
        """Pattern detection still works with word-by-word token delivery."""
        results = []
        for token in ["Hi ", "<think>", "secret", "</think>", " bye"]:
            async for r in self.aggregator.aggregate(token):
                results.append(r)

        # Handler called once when the pattern completes
        self.handler.assert_called_once()
        call_args = self.handler.call_args[0][0]
        self.assertEqual(call_args.text, "secret")

        # "Hi " yields before pattern starts, pattern is removed, " bye" yields after
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "Hi ")
        self.assertEqual(results[0].type, "token")
        self.assertEqual(results[1].text, " bye")
        self.assertEqual(results[1].type, "token")

    async def test_token_incomplete_pattern_buffers(self):
        """Incomplete pattern is buffered across calls, not leaked to output."""
        results = []
        for token in ["Hi ", "<think>", "partial"]:
            async for r in self.aggregator.aggregate(token):
                results.append(r)

        # Only "Hi " should be yielded; "<think>partial" stays buffered
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Hi ")
        self.assertEqual(results[0].type, "token")
        self.handler.assert_not_called()

    async def test_token_flush_drops_unclosed_pattern(self):
        """TOKEN mode: an unclosed REMOVE pattern is dropped on flush, not leaked."""
        results = []
        for token in ["Hi ", "<think>", "secret"]:
            async for r in self.aggregator.aggregate(token):
                results.append(r)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Hi ")

        result = await self.aggregator.flush()
        self.assertIsNone(result)
        self.handler.assert_not_called()

        # State resets after flush.
        self.assertEqual(self.aggregator.text.text, "")

    async def test_token_start_delimiter_split_across_chunks(self):
        """A start delimiter split across chunks is not leaked as plain text.

        Regression test for https://github.com/pipecat-ai/pipecat/issues/5267:
        on unmodified main, "<thin" is yielded as a token before the delimiter
        reassembles, so the REMOVE pattern is never recognized and "secret" is
        spoken.
        """
        results = []
        for token in ["Hi ", "<thin", "k>secret</think>", " bye"]:
            async for r in self.aggregator.aggregate(token):
                results.append(r)

        # The delimiter reassembles and the pattern is recognized: content is
        # stripped and the handler fires exactly once with the right content.
        self.handler.assert_called_once()
        call_args = self.handler.call_args[0][0]
        self.assertEqual(call_args.text, "secret")

        texts = [r.text for r in results]
        self.assertEqual(texts, ["Hi ", " bye"])

    async def test_token_start_delimiter_split_across_three_chunks(self):
        """A start delimiter split across three chunks reassembles correctly."""
        results = []
        for token in ["<th", "in", "k>secret</think>"]:
            async for r in self.aggregator.aggregate(token):
                results.append(r)

        self.handler.assert_called_once()
        call_args = self.handler.call_args[0][0]
        self.assertEqual(call_args.text, "secret")

        # Nothing should have leaked while the delimiter was reassembling.
        self.assertEqual(results, [])

    async def test_token_partial_delimiter_chunk_yields_nothing(self):
        """A chunk that is entirely a partial delimiter yields nothing."""
        results = [r async for r in self.aggregator.aggregate("<thin")]
        self.assertEqual(results, [])
        self.assertEqual(self.aggregator.text.text, "<thin")

    async def test_token_trailing_prefix_char_held_then_emitted(self):
        """A plain trailing '<' is held back and emitted with the next chunk once it no longer matches a delimiter."""
        results = []
        async for r in self.aggregator.aggregate("Hello <"):
            results.append(r)
        # "<" could be the start of "<think>", so it's held back.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Hello ")

        async for r in self.aggregator.aggregate(" world"):
            results.append(r)
        # Once the held-back "<" is followed by something other than "t", it
        # no longer matches a delimiter prefix and is emitted with the rest.
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1].text, "< world")

    async def test_token_split_aggregate_delimiter_recognized(self):
        """A split AGGREGATE delimiter still reassembles: content is yielded as
        its own aggregation instead of the fragments leaking as plain tokens.
        """
        code_handler = AsyncMock()
        self.aggregator.add_pattern(
            type="code_pattern",
            start_pattern="<code>",
            end_pattern="</code>",
            action=MatchAction.AGGREGATE,
        )
        self.aggregator.on_pattern_match("code_pattern", code_handler)

        results = []
        for token in ["Here is code ", "<cod", "e>pattern content</code>", " more"]:
            async for r in self.aggregator.aggregate(token):
                results.append(r)

        code_handler.assert_called_once()
        call_args = code_handler.call_args[0][0]
        self.assertEqual(call_args.text, "pattern content")

        self.assertEqual(
            [(r.type, r.text) for r in results],
            [
                ("token", "Here is code "),
                ("code_pattern", "pattern content"),
                ("token", " more"),
            ],
        )

    async def test_token_split_keep_delimiter_recognized(self):
        """A split KEEP delimiter still reassembles and triggers its handler."""
        keep_handler = AsyncMock()
        self.aggregator.add_pattern(
            type="em",
            start_pattern="<em>",
            end_pattern="</em>",
            action=MatchAction.KEEP,
        )
        self.aggregator.on_pattern_match("em", keep_handler)

        results = []
        for token in ["very <e", "m>excited</em> today"]:
            async for r in self.aggregator.aggregate(token):
                results.append(r)

        keep_handler.assert_called_once()
        call_args = keep_handler.call_args[0][0]
        self.assertEqual(call_args.text, "excited")

        # KEEP delimiters stay in the text; the split start delimiter
        # reassembles intact instead of leaking as a fragment ("<e").
        texts = [r.text for r in results]
        self.assertEqual(texts, ["very ", "<em>excited</em> today"])


if __name__ == "__main__":
    unittest.main()
