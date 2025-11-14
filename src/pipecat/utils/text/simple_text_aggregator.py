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

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


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

    @property
    def text(self) -> str:
        """Get the currently aggregated text.

        Returns:
            The text that has been accumulated in the buffer.
        """
        return self._text

    def _extract_next_sentence(self) -> Optional[str]:
        """Extract the next complete sentence from the buffer.

        Returns:
            The first complete sentence if a sentence boundary is found,
            or None if the buffer is empty or contains only incomplete text.
        """
        eos_end_marker = match_endofsentence(self._text)
        if eos_end_marker:
            # Extract the first complete sentence
            sentence = self._text[:eos_end_marker]
            # Remove it from buffer
            self._text = self._text[eos_end_marker:]
            return sentence

        return None

    async def aggregate(self, text: str) -> Optional[str]:
        """Aggregate text and return the first completed sentence.

        Adds the new text to the buffer and checks for sentence boundaries.
        When a sentence boundary is found, returns the first completed sentence
        and removes it from the buffer. Subsequent calls (even with empty strings)
        will return additional complete sentences if they exist in the buffer.

        This handles varying input patterns from different LLM providers:
        - Word-by-word tokens (e.g., 'Hello', '!', ' I', ' am', ' Doug.')
        - Chunks with one or more sentences (e.g., 'Hello! I am Doug. Nice to meet you!')

        Args:
            text: New text to add to the aggregation buffer.

        Returns:
            The first complete sentence if a sentence boundary is found,
            or None if more text is needed to complete a sentence.
        """
        self._text += text
        return self._extract_next_sentence()

    async def flush_next_sentence(self) -> Optional[str]:
        """Retrieve the next complete sentence from the buffer without adding new text.

        This method extracts the next complete sentence from the internal buffer
        without requiring new input text. It's useful for draining multiple
        complete sentences that were received in a single chunk.

        Returns:
            The next complete sentence if one exists in the buffer, or None if
            the buffer is empty or contains only incomplete text.
        """
        return self._extract_next_sentence()

    async def handle_interruption(self):
        """Handle interruptions by clearing the text buffer.

        Called when an interruption occurs in the processing pipeline,
        discarding any partially accumulated text.
        """
        self._text = ""

    async def reset(self):
        """Clear the internally aggregated text.

        Resets the aggregator to its initial empty state, discarding
        any accumulated text content.
        """
        self._text = ""
