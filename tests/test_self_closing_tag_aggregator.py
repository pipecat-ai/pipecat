#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.self_closing_tag_aggregator import SelfClosingTagAggregator


class TestSelfClosingTagAggregator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.aggregator = SelfClosingTagAggregator(["break"])

    async def test_no_tags(self):
        await self.aggregator.reset()

        # No tags involved, aggregate at end of sentence.
        result = await self.aggregator.aggregate("Hello Pipecat!")
        self.assertEqual(result, "Hello Pipecat!")
        self.assertEqual(self.aggregator.text, "")

    async def test_complete_tags(self):
        await self.aggregator.reset()

        # Complete tags, should aggregate normally.
        result = await self.aggregator.aggregate('Call us at <break time="0.1s"/>now.')
        self.assertEqual(result, 'Call us at <break time="0.1s"/>now.')
        self.assertEqual(self.aggregator.text, "")

    async def test_incomplete_tag_single_chunk(self):
        await self.aggregator.reset()

        # Incomplete tag in single chunk, should buffer.
        result = await self.aggregator.aggregate('Hello <break time="0.')
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, 'Hello <break time="0.')

    async def test_multiple_tag_types(self):
        # Test with multiple tag types
        multi_aggregator = SelfClosingTagAggregator(["break", "pause", "voice"])
        await multi_aggregator.reset()

        result = await multi_aggregator.aggregate('Hello <voice name="alice"/>world.')
        self.assertEqual(result, 'Hello <voice name="alice"/>world.')
        self.assertEqual(multi_aggregator.text, "")

        # Test incomplete with multiple types
        await multi_aggregator.reset()
        result = await multi_aggregator.aggregate('Say <pause duration="500')
        self.assertIsNone(result)
        self.assertEqual(multi_aggregator.text, 'Say <pause duration="500')

        result = await multi_aggregator.aggregate('ms"/>this slowly.')
        self.assertEqual(result, 'Say <pause duration="500ms"/>this slowly.')
        self.assertEqual(multi_aggregator.text, "")

    async def test_custom_patterns(self):
        # Test with custom regex patterns
        pattern_aggregator = SelfClosingTagAggregator(
            patterns=[r'<break\s+time="[^"]*"\s*/?>', r'<voice\s+name="[^"]*"\s*/>']
        )
        await pattern_aggregator.reset()

        # Complete custom pattern
        result = await pattern_aggregator.aggregate('Test <break time="1.5s"/> custom.')
        self.assertEqual(result, 'Test <break time="1.5s"/> custom.')
        self.assertEqual(pattern_aggregator.text, "")

        # Incomplete custom pattern
        await pattern_aggregator.reset()
        result = await pattern_aggregator.aggregate('Hello <voice name="bob')
        self.assertIsNone(result)
        self.assertEqual(pattern_aggregator.text, 'Hello <voice name="bob')

        result = await pattern_aggregator.aggregate('"/>there.')
        self.assertEqual(result, 'Hello <voice name="bob"/>there.')
        self.assertEqual(pattern_aggregator.text, "")

    async def test_sentence_boundaries_with_complete_tags(self):
        await self.aggregator.reset()

        # Multiple sentences with complete tags should split appropriately
        result = await self.aggregator.aggregate(
            'First <break time="1s"/> sentence. Second sentence.'
        )
        self.assertEqual(result, 'First <break time="1s"/> sentence.')
        self.assertEqual(self.aggregator.text, " Second sentence.")

        # Adding empty string should trigger processing of remaining complete sentence
        result = await self.aggregator.aggregate("")
        self.assertEqual(result, " Second sentence.")
        self.assertEqual(self.aggregator.text, "")

    async def test_no_sentence_ending(self):
        await self.aggregator.reset()

        # Text without sentence ending should buffer
        result = await self.aggregator.aggregate('Hello <break time="1s"/> world')
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, 'Hello <break time="1s"/> world')

    async def test_initialization_errors(self):
        # Test that initialization requires either tags or patterns
        with self.assertRaises(ValueError) as context:
            SelfClosingTagAggregator()

        self.assertIn("Must provide either 'tags' or 'patterns' parameter", str(context.exception))

        # Test that both tags and patterns work
        tag_aggregator = SelfClosingTagAggregator(["test"])
        self.assertIsNotNone(tag_aggregator)

        pattern_aggregator = SelfClosingTagAggregator(patterns=[r"<test\s*/>"])
        self.assertIsNotNone(pattern_aggregator)

    async def test_handle_interruption(self):
        await self.aggregator.reset()

        # Add some text to buffer
        result = await self.aggregator.aggregate('Buffered <break time="0.')
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, 'Buffered <break time="0.')

        # Handle interruption should clear buffer
        await self.aggregator.handle_interruption()
        self.assertEqual(self.aggregator.text, "")

    async def test_reset(self):
        await self.aggregator.reset()

        # Add some text to buffer
        result = await self.aggregator.aggregate('Some <break time="0.')
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, 'Some <break time="0.')

        # Reset should clear buffer
        await self.aggregator.reset()
        self.assertEqual(self.aggregator.text, "")

    async def test_property_access(self):
        await self.aggregator.reset()

        # Test that text property works
        self.assertEqual(self.aggregator.text, "")

        await self.aggregator.aggregate('Test <break time="0.')
        self.assertEqual(self.aggregator.text, 'Test <break time="0.')

    async def test_malformed_tags_ignored(self):
        await self.aggregator.reset()

        # Malformed tags (not matching pattern) should be ignored
        result = await self.aggregator.aggregate('Test <break_time="0.1s"/> normal.')
        self.assertEqual(result, 'Test <break_time="0.1s"/> normal.')
        self.assertEqual(self.aggregator.text, "")

    async def test_edge_case_empty_strings(self):
        await self.aggregator.reset()

        # Empty string should not cause issues
        result = await self.aggregator.aggregate("")
        self.assertIsNone(result)
        self.assertEqual(self.aggregator.text, "")

        # Add real content after empty string
        result = await self.aggregator.aggregate("Hello world.")
        self.assertEqual(result, "Hello world.")
        self.assertEqual(self.aggregator.text, "")
