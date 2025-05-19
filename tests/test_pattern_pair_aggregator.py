#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock

from pipecat.utils.text.pattern_pair_aggregator import PatternMatch, PatternPairAggregator


class TestPatternPairAggregator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.aggregator = PatternPairAggregator()
        self.test_handler = AsyncMock()

        # Add a test pattern
        self.aggregator.add_pattern_pair(
            pattern_id="test_pattern",
            start_pattern="<test>",
            end_pattern="</test>",
            remove_match=True,
        )

        # Register the mock handler
        self.aggregator.on_pattern_match("test_pattern", self.test_handler)

    async def test_pattern_match_and_removal(self):
        # First part doesn't complete the pattern
        result = await self.aggregator.aggregate("Hello <test>pattern")
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, "Hello <test>pattern")

        # Second part completes the pattern and includes an exclamation point
        result = await self.aggregator.aggregate(" content</test>!")

        # Verify the handler was called with correct PatternMatch object
        self.test_handler.assert_called_once()
        call_args = self.test_handler.call_args[0][0]
        self.assertIsInstance(call_args, PatternMatch)
        self.assertEqual(call_args.pattern_id, "test_pattern")
        self.assertEqual(call_args.full_match, "<test>pattern content</test>")
        self.assertEqual(call_args.content, "pattern content")

        # The exclamation point should be treated as a sentence boundary,
        # so the result should include just text up to and including "!"
        self.assertEqual(result, "Hello !")

        # Next sentence should be processed separately
        result = await self.aggregator.aggregate(" This is another sentence.")
        self.assertEqual(result, " This is another sentence.")

        # Buffer should be empty after returning a complete sentence
        self.assertEqual(self.aggregator.text, "")

    async def test_incomplete_pattern(self):
        # Add text with incomplete pattern
        result = await self.aggregator.aggregate("Hello <test>pattern content")

        # No complete pattern yet, so nothing should be returned
        self.assertIsNone(result)

        # The handler should not be called yet
        self.test_handler.assert_not_called()

        # Buffer should contain the incomplete text
        self.assertEqual(self.aggregator.text, "Hello <test>pattern content")

        # Reset and confirm buffer is cleared
        await self.aggregator.reset()
        self.assertEqual(self.aggregator.text, "")

    async def test_multiple_patterns(self):
        # Set up multiple patterns and handlers
        voice_handler = AsyncMock()
        emphasis_handler = AsyncMock()

        self.aggregator.add_pattern_pair(
            pattern_id="voice", start_pattern="<voice>", end_pattern="</voice>", remove_match=True
        )

        self.aggregator.add_pattern_pair(
            pattern_id="emphasis",
            start_pattern="<em>",
            end_pattern="</em>",
            remove_match=False,  # Keep emphasis tags
        )

        self.aggregator.on_pattern_match("voice", voice_handler)
        self.aggregator.on_pattern_match("emphasis", emphasis_handler)

        # Test with multiple patterns in one text block
        text = "Hello <voice>female</voice> I am <em>very</em> excited to meet you!"
        result = await self.aggregator.aggregate(text)

        # Both handlers should be called with correct data
        voice_handler.assert_called_once()
        voice_match = voice_handler.call_args[0][0]
        self.assertEqual(voice_match.pattern_id, "voice")
        self.assertEqual(voice_match.content, "female")

        emphasis_handler.assert_called_once()
        emphasis_match = emphasis_handler.call_args[0][0]
        self.assertEqual(emphasis_match.pattern_id, "emphasis")
        self.assertEqual(emphasis_match.content, "very")

        # Voice pattern should be removed, emphasis pattern should remain
        self.assertEqual(result, "Hello  I am <em>very</em> excited to meet you!")

        # Buffer should be empty
        self.assertEqual(self.aggregator.text, "")

    async def test_handle_interruption(self):
        # Start with incomplete pattern
        result = await self.aggregator.aggregate("Hello <test>pattern")
        self.assertIsNone(result)

        # Simulate interruption
        await self.aggregator.handle_interruption()

        # Buffer should be cleared
        self.assertEqual(self.aggregator.text, "")

        # Handler should not have been called
        self.test_handler.assert_not_called()

    async def test_pattern_across_sentences(self):
        # Test pattern that spans multiple sentences
        result = await self.aggregator.aggregate("Hello <test>This is sentence one.")

        # First sentence contains start of pattern but no end, so no complete pattern yet
        self.assertIsNone(result)

        # Add second part with pattern end
        result = await self.aggregator.aggregate(" This is sentence two.</test> Final sentence.")

        # Handler should be called with entire content
        self.test_handler.assert_called_once()
        call_args = self.test_handler.call_args[0][0]
        self.assertEqual(call_args.content, "This is sentence one. This is sentence two.")

        # Pattern should be removed, resulting in text with sentences merged
        self.assertEqual(result, "Hello  Final sentence.")

        # Buffer should be empty
        self.assertEqual(self.aggregator.text, "")
