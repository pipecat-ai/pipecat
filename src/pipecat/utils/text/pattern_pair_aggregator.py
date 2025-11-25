#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pattern pair aggregator for processing structured content in streaming text.

This module provides an aggregator that identifies and processes content between
pattern pairs (like XML tags or custom delimiters) in streaming text, with
support for custom handlers and configurable actions for when a pattern is found.
"""

import re
from enum import Enum
from typing import Awaitable, Callable, List, Optional, Tuple

from loguru import logger

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType, BaseTextAggregator


class MatchAction(Enum):
    """Actions to take when a pattern pair is matched.

    Parameters:
        REMOVE: The text along with its delimiters will be removed from the streaming text.
              Sentence aggregation will continue on as if this text did not exist.
        KEEP: The delimiters will be removed, but the content between them will be kept.
              Sentence aggregation will continue on with the internal text included.
        AGGREGATE: The delimiters will be removed and the content between will be treated
              as a separate aggregation. Any text before the start of the pattern will be
              returned early, whether or not a complete sentence was found. Then the pattern
              will be returned. Then the aggregation will continue on sentence matching after
              the closing delimiter is found. The content between the delimiters is not
              aggregated by sentence. It is aggregated as one single block of text.
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

    def __init__(self, content: str, type: str, full_match: str):
        """Initialize a pattern match.

        Args:
            type: The type of the matched pattern pair. It should be representative
                   of the content type (e.g., 'sentence', 'code', 'speaker', 'custom').
            full_match: The complete text including start and end patterns.
            content: The text content between the start and end patterns.
        """
        super().__init__(text=content, type=type)
        self.full_match = full_match

    def __str__(self) -> str:
        """Return a string representation of the pattern match.

        Returns:
            A descriptive string showing the pattern type and content.
        """
        return f"PatternMatch(type={self.type}, text={self.text}, full_match={self.full_match})"


class PatternPairAggregator(BaseTextAggregator):
    """Aggregator that identifies and processes content between pattern pairs.

    This aggregator buffers text until it can identify complete pattern pairs
    (defined by start and end patterns), processes the content between these
    patterns using registered handlers. By default, its aggregation method
    returns text at sentence boundaries, and remove the content found between
    any matched patterns. However, matched patterns can also be configured to
    returned as a separate aggregation object containing the content between
    their start and end patterns or left in, so that only the delimiters are
    removed and a callback can be triggered.

    This aggregator is particularly useful for processing structured content in
    streaming text, such as XML tags, markdown formatting, or custom delimiters.

    The aggregator ensures that patterns spanning multiple text chunks are
    correctly identified.
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
        stripped_text = self._text.strip()
        type = (
            pattern_start[1].get("type", AggregationType.SENTENCE)
            if pattern_start
            else AggregationType.SENTENCE
        )
        return Aggregation(text=stripped_text, type=type)

    def add_pattern(
        self,
        type: str,
        start_pattern: str,
        end_pattern: str,
        action: MatchAction = MatchAction.REMOVE,
    ) -> "PatternPairAggregator":
        """Add a pattern pair to detect in the text.

        Registers a new pattern pair with a unique identifier. The aggregator
        will look for text that starts with the start pattern and ends with
        the end pattern, and treat the content between them as a match.

        Args:
            type: Identifier for this pattern pair. Should be unique and ideally descriptive.
                  (e.g., 'code', 'speaker', 'custom'). type can not be 'sentence' or 'word' as
                  those are reserved for the default behavior.
            start_pattern: Pattern that marks the beginning of content.
            end_pattern: Pattern that marks the end of content.
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
        if type in [AggregationType.SENTENCE, AggregationType.WORD]:
            raise ValueError(
                f"The aggregation type '{type}' is reserved for default behavior and can not be used for custom patterns."
            )
        self._patterns[type] = {
            "start": start_pattern,
            "end": end_pattern,
            "type": type,
            "action": action,
        }
        return self

    def add_pattern_pair(
        self, pattern_id: str, start_pattern: str, end_pattern: str, remove_match: bool = True
    ):
        """Add a pattern pair to detect in the text.

        .. deprecated:: 0.0.95
            This function is deprecated and will be removed in a future version.
            Use `add_pattern` with a type and MatchAction instead.

            This method calls `add_pattern` setting type with the provided pattern_id and action
            to either MatchAction.REMOVE or MatchAction.KEEP based on `remove_match`.

        Args:
            pattern_id: Identifier for this pattern pair. Should be unique and ideally descriptive.
                        (e.g., 'code', 'speaker', 'custom'). pattern_id can not be 'sentence' or 'word'
                        as those arereserved for the default behavior.
            start_pattern: Pattern that marks the beginning of content.
            end_pattern: Pattern that marks the end of content.
            remove_match: If True, the matched pattern will be removed from the text. (Same as MatchAction.REMOVE)
                          If False, it will be kept and treated as normal text. (Same as MatchAction.KEEP)
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn(
                "add_pattern_pair with a pattern_id or remove_match is deprecated and will be"
                " removed in a future version. Use add_pattern with a type and MatchAction instead",
                DeprecationWarning,
                stacklevel=2,
            )

        action = MatchAction.REMOVE if remove_match else MatchAction.KEEP
        return self.add_pattern(
            type=pattern_id,
            start_pattern=start_pattern,
            end_pattern=end_pattern,
            action=action,
        )

    def on_pattern_match(
        self, type: str, handler: Callable[[PatternMatch], Awaitable[None]]
    ) -> "PatternPairAggregator":
        """Register a handler for when a pattern pair is matched.

        The handler will be called whenever a complete match for the
        specified type is found in the text.

        Args:
            type: The type of the pattern pair to trigger the handler.
            handler: Async function to call when pattern is matched.
                     The function should accept a PatternMatch object.

        Returns:
            Self for method chaining.
        """
        self._handlers[type] = handler
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

        for type, pattern_info in self._patterns.items():
            # Escape special regex characters in the patterns
            start = re.escape(pattern_info["start"])
            end = re.escape(pattern_info["end"])
            action = pattern_info["action"]

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
                    content=content.strip(), type=type, full_match=full_match
                )

                # Call the appropriate handler if registered
                if type in self._handlers:
                    try:
                        await self._handlers[type](pattern_match)
                    except Exception as e:
                        logger.error(f"Error in pattern handler for {type}: {e}")

                # Remove the pattern from the text if configured
                if action == MatchAction.REMOVE:
                    processed_text = processed_text.replace(full_match, "", 1)
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
            A tuple of (start_index, pattern_info) if an incomplete pattern is found,
            or None if no patterns are found or all patterns are complete.
        """
        for type, pattern_info in self._patterns.items():
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

        if len(patterns) > 0:
            if len(patterns) > 1:
                logger.warning(
                    f"Multiple patterns matched: {[p.type for p in patterns]}. Only the first pattern will be returned."
                )
            # If the pattern found is set to be aggregated, return it
            action = self._patterns[patterns[0].type].get("action", MatchAction.REMOVE)
            if action == MatchAction.AGGREGATE:
                self._text = ""
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
            return PatternMatch(
                content=result.strip(), type=AggregationType.SENTENCE, full_match=result
            )

        # Find sentence boundary if no incomplete patterns
        eos_marker = match_endofsentence(self._text)
        if eos_marker:
            # Extract text up to the sentence boundary
            result = self._text[:eos_marker]
            self._text = self._text[eos_marker:]
            return PatternMatch(
                content=result.strip(), type=AggregationType.SENTENCE, full_match=result
            )

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
