#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simple text aggregator for basic sentence-boundary text processing.

This module provides a straightforward text aggregator that accumulates text
until it finds an end-of-sentence marker, making it suitable for basic TTS
text processing scenarios.
"""

from typing import Optional

from pipecat.utils.string import SENTENCE_ENDING_PUNCTUATION, match_endofsentence
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType, BaseTextAggregator


class SimpleTextAggregator(BaseTextAggregator):
    """Simple text aggregator that accumulates text until sentence boundaries.

    This aggregator provides basic functionality for accumulating text tokens
    and releasing them when an end-of-sentence marker is detected. It's the
    most straightforward implementation of text aggregation for TTS processing.
    """

    def __init__(self):
        """Initialize the simple text aggregator.

        Creates an empty text buffer ready to begin accumulating text tokens.
        """
        self._text = ""
        self._needs_lookahead: bool = False

    @property
    def text(self) -> Aggregation:
        """Get the currently aggregated text.

        Returns:
            The text that has been accumulated in the buffer.
        """
        return Aggregation(text=self._text.strip(), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> Optional[Aggregation]:
        """Aggregate text and return completed sentences.

        Adds the new text to the buffer. When sentence-ending punctuation is
        detected, it waits for non-whitespace lookahead before calling NLTK.
        This prevents false positives like "$29." being detected as a sentence
        when it's actually "$29.95", and avoids unnecessary NLTK calls.

        Args:
            text: New text to add to the aggregation buffer.

        Returns:
            A complete sentence if an end-of-sentence marker is confirmed,
            or None if more text is needed.
        """
        # Add new text to buffer
        self._text += text

        return await self._check_sentence_with_lookahead(text)

    async def _check_sentence_with_lookahead(self, text: str) -> Optional[Aggregation]:
        """Check for sentence boundaries using lookahead logic.

        This method implements the core sentence detection logic with lookahead.
        When sentence-ending punctuation is detected, it waits for the next
        non-whitespace character before calling NLTK. This disambiguates cases
        like "$29." (not a sentence) vs "$29. Next" (sentence ends at period).
        Whitespace alone is not meaningful lookahead since it appears in both
        cases. Instead, the first non-whitespace character after the punctuation
        is used to confirm the sentence boundary.

        Subclasses can call this via super() to reuse the lookahead behavior
        while adding their own logic (e.g., tag handling, pattern matching).

        Args:
            text: The most recently added text (used for lookahead check).

        Returns:
            Aggregation if sentence found, None otherwise.
        """
        # If we need lookahead, check if we now have non-whitespace
        if self._needs_lookahead:
            # Check if the new character is non-whitespace
            if text.strip():
                # We have meaningful lookahead, call NLTK
                self._needs_lookahead = False
                eos_marker = match_endofsentence(self._text)

                if eos_marker:
                    # NLTK confirmed a sentence - return it
                    result = self._text[:eos_marker]
                    self._text = self._text[eos_marker:]
                    return Aggregation(text=result, type=AggregationType.SENTENCE)
                # No sentence found - keep accumulating
                return None
            # Still whitespace, keep waiting
            return None

        # Check if we just added sentence-ending punctuation
        if self._text[-1] in SENTENCE_ENDING_PUNCTUATION:
            # Mark that we need lookahead (don't call NLTK yet)
            self._needs_lookahead = True

        return None

    async def flush(self) -> Optional[Aggregation]:
        """Flush any remaining text in the buffer.

        Returns any text remaining in the buffer. This is called at the end
        of a stream to ensure all text is processed.

        Returns:
            Any remaining text as a sentence, or None if buffer is empty.
        """
        if self._text:
            # Return whatever we have in the buffer
            result = self._text
            await self.reset()
            return Aggregation(text=result.strip(), type=AggregationType.SENTENCE)
        return None

    async def handle_interruption(self):
        """Handle interruptions by clearing the text buffer.

        Called when an interruption occurs in the processing pipeline,
        discarding any partially accumulated text.
        """
        self._text = ""
        self._needs_lookahead = False

    async def reset(self):
        """Clear the internally aggregated text.

        Resets the aggregator to its initial empty state, discarding
        any accumulated text content.
        """
        self._text = ""
        self._needs_lookahead = False
