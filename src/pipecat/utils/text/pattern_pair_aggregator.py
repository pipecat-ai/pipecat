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
from enum import Enum
from typing import Awaitable, Callable, List, Optional, Tuple

from loguru import logger

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import Aggregation, BaseTextAggregator


class MatchAction(Enum):
    """Actions to take when a pattern pair is matched.

    Parameters:
        REMOVE: Remove the matched pattern from the text.
        KEEP: Keep the matched pattern in the text as normal text.
        AGGREGATE: Return the matched pattern as a separate aggregation object.
    """

    REMOVE = "remove"
    KEEP = "keep"
    AGGREGATE = "aggregate"


class PatternMatch(Aggregation):
    """Represents a matched pattern pair with its content.

    A PatternMatch object is created when a complete pattern pair is found
    in the text. It contains information about which pattern was matched,
    the full matched text (including start and end patterns), and the
    content between the patterns.
    """

    def __init__(self, pattern_id: str, full_match: str, content: str, type: str):
        """Initialize a pattern match.

        Args:
            pattern_id: The identifier of the matched pattern pair.
            full_match: The complete text including start and end patterns.
            content: The text content between the start and end patterns.
            type: The type of aggregation the matched content represents
                  (e.g., 'code', 'speaker', 'custom').
        """
        super().__init__(text=content, type=type)
        self.pattern_id = pattern_id
        self.full_match = full_match

    def __str__(self) -> str:
        """Return a string representation of the pattern match.

        Returns:
            A descriptive string showing the pattern ID and content.
        """
        return f"PatternMatch(id={self.pattern_id}, content={self.text}, full_match={self.full_match}, type={self.type})"


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

    def __init__(self, **kwargs):
        """Initialize the pattern pair aggregator.

        Creates an empty aggregator with no patterns or handlers registered.
        Text buffering and pattern detection will begin when text is aggregated.
        """
        self._text = ""
        self._patterns = {}
        self._handlers = {}

    @property
    def text(self) -> Aggregation:
        """Get the currently aggregated text.

        Returns:
            The text that has been accumulated in the buffer.
        """
        pattern_start = self._match_start_of_pattern(self._text)
        if pattern_start:
            return Aggregation(self._text, pattern_start[1].get("type", "sentence"))
        return Aggregation(self._text, "sentence")

    def add_pattern_pair(
        self,
        pattern_id: str,
        start_pattern: str,
        end_pattern: str,
        type: str,
        action: MatchAction = MatchAction.REMOVE,
    ) -> "PatternPairAggregator":
        """Add a pattern pair to detect in the text.

        Registers a new pattern pair with a unique identifier. The aggregator
        will look for text that starts with the start pattern and ends with
        the end pattern, and treat the content between them as a match.

        Args:
            pattern_id: Unique identifier for this pattern pair.
            start_pattern: Pattern that marks the beginning of content.
            end_pattern: Pattern that marks the end of content.
            type: The type of aggregation the matched content represents
                  (e.g., 'code', 'speaker', 'custom').
            action: What to do when a complete pattern is matched:
                    - MatchAction.REMOVE: Remove the matched pattern from the text.
                    - MatchAction.KEEP: Keep the matched pattern in the text and treat it as
                                        normal text. This allows you to register handlers for
                                        the pattern without affecting the aggregation logic.
                    - MatchAction.AGGREGATE: Return the matched pattern as a separate
                                             aggregation object.

        Returns:
            Self for method chaining.
        """
        self._patterns[pattern_id] = {
            "start": start_pattern,
            "end": end_pattern,
            "type": type,
            "action": action,
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

    async def _process_complete_patterns(self, text: str) -> Tuple[List[PatternMatch], str]:
        """Process all complete pattern pairs in the text.

        Searches for all complete pattern pairs in the text, calls the
        appropriate handlers, and optionally removes the matches.

        Args:
            text: The text to process for pattern matches.

        Returns:
            Tuple of (all_matches, processed_text) where:

            - all_matches is a list of all pattern matches found. Note: There really should only ever be 1.
            - processed_text is the text after processing patterns. If no patterns are found, it will be the same as input text.
        """
        all_matches = []
        processed_text = text

        for pattern_id, pattern_info in self._patterns.items():
            # Escape special regex characters in the patterns
            start = re.escape(pattern_info["start"])
            end = re.escape(pattern_info["end"])
            action = pattern_info["action"]
            match_type = pattern_info["type"]

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
                    pattern_id=pattern_id, full_match=full_match, content=content, type=match_type
                )

                # Call the appropriate handler if registered
                if pattern_id in self._handlers:
                    try:
                        await self._handlers[pattern_id](pattern_match)
                    except Exception as e:
                        logger.error(f"Error in pattern handler for {pattern_id}: {e}")

                # Remove the pattern from the text if configured
                if action == MatchAction.REMOVE:
                    processed_text = processed_text.replace(full_match, "", 1)
                    # modified = True
                else:
                    all_matches.append(pattern_match)

        return all_matches, processed_text

    def _match_start_of_pattern(self, text: str) -> Optional[Tuple[int, dict]]:
        """Check if text contains incomplete pattern pairs.

        Determines whether the text contains any start patterns without
        matching end patterns, which would indicate incomplete content.

        Args:
            text: The text to check for incomplete patterns.

        Returns:
            A tuple of (start_index, type) if an incomplete pattern is found,
            or None if no patterns are found or all patterns are complete.
        """
        for pattern_id, pattern_info in self._patterns.items():
            start = pattern_info["start"]
            end = pattern_info["end"]

            # Count occurrences
            start_count = text.count(start)
            end_count = text.count(end)

            # If there are more starts than ends, we have incomplete patterns
            # Again, this is written generically but there only ever should
            # be one pattern active at a time, so the counts should be 0 or 1.
            # Which is why we base the return on the first found.
            if start_count > end_count:
                start_index = text.find(start)
                return [start_index, pattern_info]

        return None

    async def aggregate(self, text: str) -> Optional[PatternMatch]:
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
        patterns, processed_text = await self._process_complete_patterns(self._text)

        self._text = processed_text

        #
        if len(patterns) > 0:
            if len(patterns) > 1:
                logger.warning(
                    f"Multiple patterns matched: {[p.pattern_id for p in patterns]}. Only the first pattern will be returned."
                )
            # If the pattern found is set to be aggregated, return it
            action = self._patterns[patterns[0].pattern_id].get("action", MatchAction.REMOVE)
            if action == MatchAction.AGGREGATE:
                self._text = ""
                print(f"Returning pattern: {patterns[0]}")
                return patterns[0]

        # Check if we have incomplete patterns
        pattern_start = self._match_start_of_pattern(self._text)
        if pattern_start is not None:
            # If the start pattern is at the beginning or should not be separately aggregated, return None
            if (
                pattern_start[0] == 0
                or pattern_start[1].get("action", MatchAction.REMOVE) != MatchAction.AGGREGATE
            ):
                return None
            # Otherwise, strip the text up to the start pattern and return it
            result = self._text[: pattern_start[0]]
            self._text = self._text[pattern_start[0] :]
            return PatternMatch(f"_sentence", result, result, "sentence")

        # Find sentence boundary if no incomplete patterns
        eos_marker = match_endofsentence(self._text)
        if eos_marker:
            # Extract text up to the sentence boundary
            result = self._text[:eos_marker]
            self._text = self._text[eos_marker:]
            return PatternMatch(f"_sentence", result, result, "sentence")

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
