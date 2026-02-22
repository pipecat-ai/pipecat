#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Skip tags aggregator for preventing sentence boundaries within tagged content.

This module provides a text aggregator that prevents end-of-sentence matching
between specified start/end tag pairs, ensuring that tagged content is processed
as a unit regardless of internal punctuation.
"""

from typing import AsyncIterator, Optional, Sequence

from pipecat.utils.string import StartEndTags, parse_start_end_tags
from pipecat.utils.text.base_text_aggregator import Aggregation
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class SkipTagsAggregator(SimpleTextAggregator):
    """Aggregator that prevents end of sentence matching between start/end tags.

    This aggregator buffers text until it finds an end of sentence or a start
    tag. If a start tag is found the aggregator will keep aggregating text
    unconditionally until the corresponding end tag is found. It's particularly
    useful for processing content with custom delimiters that should prevent
    text from being considered for end of sentence matching.

    The aggregator ensures that tags spanning multiple text chunks are correctly
    identified and that content within tags is never split at sentence boundaries.
    """

    def __init__(self, tags: Sequence[StartEndTags]):
        """Initialize the skip tags aggregator.

        Args:
            tags: Sequence of StartEndTags objects defining the tag pairs
                  that should prevent sentence boundary detection.
        """
        super().__init__()
        self._tags = tags
        self._current_tag: Optional[StartEndTags] = None
        self._current_tag_index: int = 0

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Aggregate text while respecting tag boundaries.

        Processes the input text character-by-character, updates tag state, and
        uses the parent's lookahead logic for sentence detection when not
        inside tags.

        Args:
            text: Text to aggregate.

        Yields:
            Aggregation objects containing text up to a sentence boundary,
            marked as SENTENCE type.
        """
        # Process text character by character
        for char in text:
            self._text += char

            # Update tag state
            (self._current_tag, self._current_tag_index) = parse_start_end_tags(
                self._text, self._tags, self._current_tag, self._current_tag_index
            )

            # If inside tags, don't check for sentences
            if self._current_tag:
                continue

            # Otherwise, use parent's lookahead logic for sentence detection
            result = await super()._check_sentence_with_lookahead(char)
            if result:
                yield result

    async def handle_interruption(self):
        """Handle interruptions by clearing the buffer and tag state.

        Called when an interruption occurs in the processing pipeline,
        to reset the state and discard any partially aggregated text.
        """
        await super().handle_interruption()
        self._current_tag = None
        self._current_tag_index = 0

    async def reset(self):
        """Clear the internally aggregated text and tag state.

        Resets the aggregator to its initial state, discarding any
        buffered text.
        """
        await super().reset()
        self._current_tag = None
        self._current_tag_index = 0
