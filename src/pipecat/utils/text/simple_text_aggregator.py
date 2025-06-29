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

    async def aggregate(self, text: str) -> Optional[str]:
        """Aggregate text and return completed sentences.

        Adds the new text to the buffer and checks for end-of-sentence markers.
        When a sentence boundary is found, returns the completed sentence and
        removes it from the buffer.

        Args:
            text: New text to add to the aggregation buffer.

        Returns:
            A complete sentence if an end-of-sentence marker is found,
            or None if more text is needed to complete a sentence.
        """
        result: Optional[str] = None

        self._text += text

        eos_end_marker = match_endofsentence(self._text)
        if eos_end_marker:
            result = self._text[:eos_end_marker]
            self._text = self._text[eos_end_marker:]

        return result

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
