#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pattern pair aggregator for processing structured content in streaming text.

This module provides an aggregator that identifies and processes content between
pattern pairs (like XML tags or custom delimiters) in streaming text, with
support for custom handlers and configurable pattern removal.
"""

import re
from typing import Awaitable, Callable, Optional, Tuple

from loguru import logger

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


class PatternMatch:
    """Represents a matched pattern pair with its content.

    A PatternMatch object is created when a complete pattern pair is found
    in the text. It contains information about which pattern was matched,
    the full matched text (including start and end patterns), and the
    content between the patterns.
    """

    def __init__(self, pattern_id: str, full_match: str, content: str):
        """Initialize a pattern match.

        Args:
            pattern_id: The identifier of the matched pattern pair.
            full_match: The complete text including start and end patterns.
            content: The text content between the start and end patterns.
        """
        self.pattern_id = pattern_id
        self.full_match = full_match
        self.content = content

    def __str__(self) -> str:
        """Return a string representation of the pattern match.

        Returns:
            A descriptive string showing the pattern ID and content.
        """
        return f"PatternMatch(id={self.pattern_id}, content={self.content})"


class PatternPairAggregator(BaseTextAggregator):
    """Aggregator that identifies and processes content between pattern pairs.

    This aggregator buffers text until it can identify complete pattern pairs
    (defined by start and end patterns), processes the content between these
    patterns using registered handlers, and returns text at sentence boundaries.
    It's particularly useful for processing structured content in streaming text,
    such as XML tags, markdown formatting, or custom delimiters.

    The aggregator ensures that patterns spanning multiple text chunks are
    correctly identified and handles cases where patterns contain sentence
    boundaries.
    """

    def __init__(self):
        """Initialize the pattern pair aggregator.

        Creates an empty aggregator with no patterns or handlers registered.
        Text buffering and pattern detection will begin when text is aggregated.
        """
        self._text = ""
        self._patterns = {}
        self._handlers = {}

    @property
    def text(self) -> str:
        """Get the currently buffered text.

        Returns:
            The current text buffer content that hasn't been processed yet.
        """
        return self._text

    def add_pattern_pair(
        self, pattern_id: str, start_pattern: str, end_pattern: str, remove_match: bool = True
    ) -> "PatternPairAggregator":
        """Add a pattern pair to detect in the text.

        Registers a new pattern pair with a unique identifier. The aggregator
        will look for text that starts with the start pattern and ends with
        the end pattern, and treat the content between them as a match.

        Args:
            pattern_id: Unique identifier for this pattern pair.
            start_pattern: Pattern that marks the beginning of content.
            end_pattern: Pattern that marks the end of content.
            remove_match: Whether to remove the matched content from the text.

        Returns:
            Self for method chaining.
        """
        self._patterns[pattern_id] = {
            "start": start_pattern,
            "end": end_pattern,
            "remove_match": remove_match,
        }
        return self

    def on_pattern_match(
        self, pattern_id: str, handler: Callable[[PatternMatch], Awaitable[None]]
    ) -> "PatternPairAggregator":
        """Register a handler for when a pattern pair is matched.

        The handler will be called whenever a complete match for the
        specified pattern ID is found in the text.

        Args:
            pattern_id: ID of the pattern pair to match.
            handler: Async function to call when pattern is matched.
                     The function should accept a PatternMatch object.

        Returns:
            Self for method chaining.
        """
        self._handlers[pattern_id] = handler
        return self

    async def _process_complete_patterns(self, text: str) -> Tuple[str, bool]:
        """Process all complete pattern pairs in the text.

        Searches for all complete pattern pairs in the text, calls the
        appropriate handlers, and optionally removes the matches.

        Args:
            text: The text to process for pattern matches.

        Returns:
            Tuple of (processed_text, was_modified) where:

            - processed_text is the text after processing patterns
            - was_modified indicates whether any changes were made
        """
        processed_text = text
        modified = False

        for pattern_id, pattern_info in self._patterns.items():
            # Escape special regex characters in the patterns
            start = re.escape(pattern_info["start"])
            end = re.escape(pattern_info["end"])
            remove_match = pattern_info["remove_match"]

            # Create regex to match from start pattern to end pattern
            # The .*? is non-greedy to handle nested patterns
            regex = f"{start}(.*?){end}"

            # Find all matches
            match_iter = re.finditer(regex, processed_text, re.DOTALL)
            matches = list(match_iter)  # Convert to list for safe iteration

            for match in matches:
                content = match.group(1)  # Content between patterns
                full_match = match.group(0)  # Full match including patterns

                # Create pattern match object
                pattern_match = PatternMatch(
                    pattern_id=pattern_id, full_match=full_match, content=content
                )

                # Call the appropriate handler if registered
                if pattern_id in self._handlers:
                    try:
                        await self._handlers[pattern_id](pattern_match)
                    except Exception as e:
                        logger.error(f"Error in pattern handler for {pattern_id}: {e}")

                # Remove the pattern from the text if configured
                if remove_match:
                    processed_text = processed_text.replace(full_match, "", 1)
                    modified = True

        return processed_text, modified

    def _has_incomplete_patterns(self, text: str) -> bool:
        """Check if text contains incomplete pattern pairs.

        Determines whether the text contains any start patterns without
        matching end patterns, which would indicate incomplete content.

        Args:
            text: The text to check for incomplete patterns.

        Returns:
            True if there are incomplete patterns, False otherwise.
        """
        for pattern_id, pattern_info in self._patterns.items():
            start = pattern_info["start"]
            end = pattern_info["end"]

            # Count occurrences
            start_count = text.count(start)
            end_count = text.count(end)

            # If there are more starts than ends, we have incomplete patterns
            if start_count > end_count:
                return True

        return False

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

        # Process any complete patterns in the buffer
        processed_text, modified = await self._process_complete_patterns(self._text)

        # Only update the buffer if modifications were made
        if modified:
            self._text = processed_text

        # Check if we have incomplete patterns
        if self._has_incomplete_patterns(self._text):
            # Still waiting for complete patterns
            return None

        # Find sentence boundary if no incomplete patterns
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
        buffered text and clearing pattern tracking state.
        """
        self._text = ""
