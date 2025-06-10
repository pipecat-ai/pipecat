#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional, Sequence

from pipecat.utils.string import StartEndTags, match_endofsentence, parse_start_end_tags
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


class SkipTagsAggregator(BaseTextAggregator):
    """Aggregator that prevents end of sentence matching between start/end tags.

    This aggregator buffers text until it finds an end of sentence or a start
    tag. If a start tag is found the aggregator will keep aggregating text
    unconditionally until the corresponding end tag is found. It's particularly
    useful for processing content with custom delimiters that should prevent
    text from being considered for end of sentence matching..

    The aggregator ensures that tags spanning multiple text chunks are correctly
    identified.

    """

    def __init__(self, tags: Sequence[StartEndTags]):
        """Initialize the pattern pair aggregator.

        Creates an empty aggregator with no patterns or handlers registered.
        """
        self._text = ""
        self._tags = tags
        self._current_tag: Optional[StartEndTags] = None
        self._current_tag_index: int = 0

    @property
    def text(self) -> str:
        """Get the currently buffered text.

        Returns:
            The current text buffer content.
        """
        return self._text

    async def aggregate(self, text: str) -> Optional[str]:
        """Aggregate text and process pattern pairs.

        This method adds the new text to the buffer, processes any complete pattern
        pairs, and returns processed text up to sentence boundaries if possible.
        If there are incomplete patterns (start without matching end), it will
        continue buffering text.

        Args:
            text: New text to add to the buffer.

        Returns:
            Processed text up to a sentence boundary, or None if more
            text is needed to form a complete sentence or pattern.
        """
        # Add new text to buffer
        self._text += text

        (self._current_tag, self._current_tag_index) = parse_start_end_tags(
            self._text, self._tags, self._current_tag, self._current_tag_index
        )

        # Find sentence boundary if no incomplete patterns
        if not self._current_tag:
            eos_marker = match_endofsentence(self._text)
            if eos_marker:
                # Extract text up to the sentence boundary
                result = self._text[:eos_marker]
                self._text = self._text[eos_marker:]
                return result

        # No complete sentence found yet
        return None

    async def handle_interruption(self):
        """Handle interruptions by clearing the buffer.

        Called when an interruption occurs in the processing pipeline,
        to reset the state and discard any partially aggregated text.
        """
        self._text = ""

    async def reset(self):
        """Clear the internally aggregated text.

        Resets the aggregator to its initial state, discarding any
        buffered text.
        """
        self._text = ""
