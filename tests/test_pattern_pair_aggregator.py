#
# Copyright (c) 2024-2025 Daily
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
        self.aggregator.add_pattern_pair(
            pattern_id="test_pattern",
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
        # Feed text character by character
        full_text = "Hello <test>pattern content</test>!"
        result = None

        for char in full_text:
            result = await self.aggregator.aggregate(char)
            if result:
                break

        # Verify the handler was called with correct PatternMatch object
        self.test_handler.assert_called_once()
        call_args = self.test_handler.call_args[0][0]
        self.assertIsInstance(call_args, PatternMatch)
        self.assertEqual(call_args.type, "test_pattern")
        self.assertEqual(call_args.full_match, "<test>pattern content</test>")
        self.assertEqual(call_args.text, "pattern content")

        # The exclamation point should be treated as a sentence boundary,
        # but with lookahead, we need more text or flush() to confirm
        self.assertIsNone(result)  # Waiting for lookahead after "!"

        # Next sentence should provide the lookahead and trigger the previous sentence
        for char in " This is another sentence.":
            result = await self.aggregator.aggregate(char)
            if result:
                break

        self.assertEqual(result.text, "Hello !")
        self.assertEqual(result.type, "sentence")

        # Now flush to get the remaining sentence
        result = await self.aggregator.flush()
        self.assertEqual(result.text, "This is another sentence.")

        # Buffer should be empty after returning a complete sentence
        self.assertEqual(self.aggregator.text.text, "")

    async def test_pattern_match_and_aggregate(self):
        # Feed text character by character until we get the first aggregation
        result = None
        for char in "Here is code <code>pattern content</code>":
            result = await self.aggregator.aggregate(char)
            if result:
                break

        # First result should be "Here is code" when pattern starts
        self.assertEqual(result.text, "Here is code")
        self.assertEqual(self.aggregator.text.text, "<code>pattern content</code>")
        self.assertEqual(self.aggregator.text.type, "code_pattern")

        # Continue feeding to complete the pattern
        result = None
        for char in "":  # Already fed all chars, just need to check state
            result = await self.aggregator.aggregate(char)
            if result:
                break

        # Since we already fed all characters, we need to trigger pattern completion
        # by checking what's in the buffer - the pattern should have been processed
        # Let's continue with a space to trigger the next check
        result = await self.aggregator.aggregate(" ")

        # Verify the handler was called with correct PatternMatch object
        self.code_handler.assert_called_once()
        call_args = self.code_handler.call_args[0][0]
        self.assertIsInstance(call_args, PatternMatch)
        self.assertEqual(call_args.type, "code_pattern")
        self.assertEqual(call_args.full_match, "<code>pattern content</code>")
        self.assertEqual(call_args.text, "pattern content")
        self.assertEqual(result.text, "pattern content")
        self.assertEqual(result.type, "code_pattern")

        # Next sentence should be processed separately
        # With lookahead, we need to flush to get the final sentence
        for char in "This is another sentence.":
            result = await self.aggregator.aggregate(char)
            if result:
                break

        self.assertIsNone(result)  # Waiting for lookahead after "."

        result = await self.aggregator.flush()
        self.assertEqual(result.text, "This is another sentence.")
        self.assertEqual(result.type, "sentence")

        # Buffer should be empty after returning a complete sentence
        self.assertEqual(self.aggregator.text.text, "")

    async def test_incomplete_pattern(self):
        # Feed text character by character with incomplete pattern
        result = None
        for char in "Hello <test>pattern content":
            result = await self.aggregator.aggregate(char)
            if result:
                break

        # No complete pattern yet, so nothing should be returned
        self.assertIsNone(result)

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

        # Feed text character by character
        text = "Hello <voice>female</voice> I am <em>very</em> excited to meet you!"
        result = None
        for char in text:
            result = await self.aggregator.aggregate(char)
            if result:
                break

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
        self.assertIsNone(result)  # Waiting for lookahead after "!"

        result = await self.aggregator.flush()
        # Voice pattern should be removed, emphasis pattern should remain
        self.assertEqual(result.text, "Hello  I am <em>very</em> excited to meet you!")

        # Buffer should be empty
        self.assertEqual(self.aggregator.text.text, "")

    async def test_handle_interruption(self):
        # Feed text character by character with incomplete pattern
        result = None
        for char in "Hello <test>pattern":
            result = await self.aggregator.aggregate(char)
            if result:
                break

        self.assertIsNone(result)

        # Simulate interruption
        await self.aggregator.handle_interruption()

        # Buffer should be cleared
        self.assertEqual(self.aggregator.text.text, "")

        # Handler should not have been called
        self.test_handler.assert_not_called()

    async def test_pattern_across_sentences(self):
        # Feed text character by character - pattern spans multiple sentences
        full_text = "Hello <test>This is sentence one. This is sentence two.</test> Final sentence."
        result = None

        for char in full_text:
            result = await self.aggregator.aggregate(char)
            if result:
                break

        # Handler should be called with entire content
        self.test_handler.assert_called_once()
        call_args = self.test_handler.call_args[0][0]
        self.assertEqual(call_args.text, "This is sentence one. This is sentence two.")

        # With lookahead, we need to flush to get the final sentence
        self.assertIsNone(result)  # Waiting for lookahead after "."

        result = await self.aggregator.flush()
        # Pattern should be removed, resulting in text with sentences merged
        self.assertEqual(result.text, "Hello  Final sentence.")

        # Buffer should be empty
        self.assertEqual(self.aggregator.text.text, "")
