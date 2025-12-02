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
        # Feed character-by-character
        for char in "Hello ":
            assert await self.aggregator.aggregate(char) == None
        assert self.aggregator.text.text == "Hello"
        await self.aggregator.reset()
        assert self.aggregator.text.text == ""

    async def test_simple_sentence(self):
        # Feed character-by-character: "Hello Pipecat!"
        for char in "Hello Pipecat":
            assert await self.aggregator.aggregate(char) == None
        # After "!", lookahead waits for confirmation
        assert await self.aggregator.aggregate("!") == None
        # Flush to get the pending sentence
        aggregate = await self.aggregator.flush()
        assert aggregate.text == "Hello Pipecat!"
        assert aggregate.type == "sentence"
        assert self.aggregator.text.text == ""

    async def test_multiple_sentences(self):
        # Feed character-by-character: "Hello Pipecat! How are you?"
        for char in "Hello Pipecat":
            assert await self.aggregator.aggregate(char) == None
        # Hit "!" - wait for lookahead
        assert await self.aggregator.aggregate("!") == None
        # Space is whitespace - keep waiting
        assert await self.aggregator.aggregate(" ") == None
        # "H" confirms sentence end
        result = await self.aggregator.aggregate("H")
        assert result.text == "Hello Pipecat!"
        # Continue with second sentence
        for char in "ow are you":
            assert await self.aggregator.aggregate(char) == None
        # Hit "?" - wait for lookahead
        assert await self.aggregator.aggregate("?") == None
        # Flush to get the pending sentence
        result = await self.aggregator.flush()
        assert result.text == "How are you?"

    async def test_lookahead_decimal_number(self):
        """Test that $29.95 is not split at $29."""
        # Feed character by character: "Ask me for only $29.95/month."
        for char in "Ask me for only $29":
            assert await self.aggregator.aggregate(char) == None
        # When we hit ".", it looks like end of sentence, but should wait for lookahead
        assert await self.aggregator.aggregate(".") == None
        # Next character "9" confirms it's not end of sentence (NLTK changes boundary)
        for char in "95/month":
            assert await self.aggregator.aggregate(char) == None
        # Now we hit the real end of sentence - wait for lookahead
        assert await self.aggregator.aggregate(".") == None
        # Can use flush() to get the pending sentence at end of stream
        result = await self.aggregator.flush()
        assert result.text == "Ask me for only $29.95/month."

    async def test_lookahead_abbreviation(self):
        """Test that Mr. Smith is not split at Mr."""
        # Feed character by character: "Hello Mr. Smith."
        for char in "Hello Mr":
            assert await self.aggregator.aggregate(char) == None
        # When we hit ".", it looks like end of sentence, but should wait for lookahead
        assert await self.aggregator.aggregate(".") == None
        # Space alone is not enough
        assert await self.aggregator.aggregate(" ") == None
        # "S" confirms it's not end of sentence (NLTK changes boundary detection)
        for char in "Smith":
            assert await self.aggregator.aggregate(char) == None
        # Now we hit the real end of sentence - wait for lookahead
        assert await self.aggregator.aggregate(".") == None
        # Can use flush() to get the pending sentence at end of stream
        result = await self.aggregator.flush()
        assert result.text == "Hello Mr. Smith."

    async def test_lookahead_actual_sentence_end(self):
        """Test that a real sentence end is detected after lookahead."""
        # Feed character by character: "Hello world. Next sentence"
        for char in "Hello world":
            assert await self.aggregator.aggregate(char) == None
        # Hit period - should wait for lookahead
        assert await self.aggregator.aggregate(".") == None
        # Space alone is not enough - need non-whitespace for meaningful lookahead
        assert await self.aggregator.aggregate(" ") == None
        # Capital letter confirms sentence end (NLTK detects boundary at same position)
        result = await self.aggregator.aggregate("N")
        assert result.text == "Hello world."
        # Continue with next sentence
        assert await self.aggregator.aggregate("e") == None

    async def test_flush_pending_sentence(self):
        """Test that flush() returns pending sentence waiting for lookahead."""
        # Feed up to a period
        for char in "Hello world":
            assert await self.aggregator.aggregate(char) == None
        assert await self.aggregator.aggregate(".") == None
        # At this point, "Hello world." is pending lookahead
        # Call flush to get it
        result = await self.aggregator.flush()
        assert result is not None
        assert result.text == "Hello world."
        # Flush again should return None
        assert await self.aggregator.flush() == None

    async def test_flush_with_no_pending(self):
        """Test that flush() returns any remaining text in buffer."""
        assert await self.aggregator.aggregate("Hello") == None
        result = await self.aggregator.flush()
        # flush() now returns any remaining text, not just pending lookahead
        assert result is not None
        assert result.text == "Hello"
        # Buffer should be empty after flush
        assert self.aggregator.text.text == ""

    async def test_flush_after_lookahead_confirmed(self):
        """Test flush after lookahead has already confirmed sentence."""
        for char in "Hello.":
            await self.aggregator.aggregate(char)
        # Space alone is not enough - still waiting
        assert await self.aggregator.aggregate(" ") == None
        # Non-whitespace lookahead confirms it's a sentence
        result = await self.aggregator.aggregate("W")
        assert result.text == "Hello."
        # flush() returns any remaining text (the "W" in this case)
        result = await self.aggregator.flush()
        assert result.text == "W"
