#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pattern pair aggregator for processing structured content in streaming text.

This module provides an aggregator that identifies and processes content between
pattern pairs (like XML tags or custom delimiters) in streaming text, with
support for custom handlers and configurable actions for when a pattern is found.
"""

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import Enum

from loguru import logger

from pipecat.utils.string import longest_trailing_partial_match
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


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


class PatternPairAggregator(SimpleTextAggregator):
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

        Args:
            **kwargs: Additional arguments passed to SimpleTextAggregator (e.g. aggregation_type).
        """
        super().__init__(**kwargs)
        self._patterns = {}
        self._handlers = {}
        self._last_processed_position = 0  # Track where we last checked for complete patterns

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
            action: What to do when a complete pattern is matched.

                - MatchAction.REMOVE: Remove the matched pattern from the text.
                - MatchAction.KEEP: Keep the matched pattern in the text and treat it as normal text. This allows you to register handlers for the pattern without affecting the aggregation logic.
                - MatchAction.AGGREGATE: Return the matched pattern as a separate aggregation object.

        Returns:
            Self for method chaining.
        """
        if type in [AggregationType.SENTENCE, AggregationType.WORD, AggregationType.TOKEN]:
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

    async def _process_complete_patterns(
        self, text: str, last_processed_position: int = 0
    ) -> tuple[list[PatternMatch], str]:
        """Process newly complete pattern pairs in the text.

        Searches for pattern pairs that have been completed since last_processed_position,
        calls the appropriate handlers, and optionally removes the matches.

        Args:
            text: The text to process for pattern matches.
            last_processed_position: The position in text that was already processed.
                Only patterns that end at or after this position will be processed.

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

                # Check if this pattern was already processed
                already_processed = match.end() <= last_processed_position

                # Only call handler for newly completed patterns
                if not already_processed and type in self._handlers:
                    try:
                        await self._handlers[type](pattern_match)
                    except Exception as e:
                        logger.error(f"Error in pattern handler for {type}: {e}")

                # Handle pattern based on action
                if action == MatchAction.REMOVE:
                    # Remove patterns are only removed once (when newly completed)
                    if not already_processed:
                        processed_text = processed_text.replace(full_match, "", 1)
                else:
                    # KEEP/AGGREGATE patterns stay in all_matches
                    all_matches.append(pattern_match)

        return all_matches, processed_text

    def _match_start_of_pattern(self, text: str) -> tuple[int, dict] | None:
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
                return (start_index, pattern_info)

        return None

    async def aggregate(self, text: str) -> AsyncIterator[PatternMatch]:
        """Aggregate text and process pattern pairs.

        Processes the input text character-by-character, handles pattern pairs,
        and uses the parent's lookahead logic for sentence detection when no
        patterns are active.

        In TOKEN mode, pattern detection still works but non-pattern text is
        yielded as TOKEN aggregations instead of waiting for sentence boundaries.
        Text ending in a partial start delimiter (e.g. ``<thin`` of
        ``<think>``) is held back until a later chunk determines whether it
        begins a pattern, so a delimiter split across chunks is still
        recognized.

        Args:
            text: Text to aggregate.

        Yields:
            PatternMatch objects as patterns complete or sentences are detected.
        """
        # Process text character by character
        for char in text:
            self._text += char

            # Process any newly complete patterns in the buffer
            # Only patterns that complete after _last_processed_position will trigger handlers
            patterns, processed_text = await self._process_complete_patterns(
                self._text, self._last_processed_position
            )

            # Update the last processed position to prevent re-processing patterns
            # This tracks where in the buffer we've already called handlers, so we
            # only trigger handlers once when a pattern completes
            self._last_processed_position = len(self._text)

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
                    yield patterns[0]
                    continue

            # Check if we have incomplete patterns
            pattern_start = self._match_start_of_pattern(self._text)
            if pattern_start is not None:
                # If the start pattern is at the beginning or should not be separately aggregated, continue
                if (
                    pattern_start[0] == 0
                    or pattern_start[1].get("action", MatchAction.REMOVE) != MatchAction.AGGREGATE
                ):
                    continue
                # For AGGREGATE patterns: yield any text before the pattern starts
                # This ensures text doesn't get stuck in the buffer waiting for sentence
                # boundaries when a pattern begins (e.g., "Here is code <code>..." yields "Here is code")
                result = self._text[: pattern_start[0]]
                self._text = self._text[pattern_start[0] :]
                agg_type = (
                    AggregationType.TOKEN
                    if self._aggregation_type == AggregationType.TOKEN
                    else AggregationType.SENTENCE
                )
                yield PatternMatch(content=result.strip(), type=agg_type, full_match=result)
                continue

            if self._aggregation_type != AggregationType.TOKEN:
                # Use parent's lookahead logic for sentence detection
                aggregation = await super()._check_sentence_with_lookahead(char)
                if aggregation:
                    # Convert to PatternMatch for consistency with return type
                    yield PatternMatch(
                        content=aggregation.text,
                        type=aggregation.type,
                        full_match=aggregation.text,
                    )

        # In TOKEN mode, yield any accumulated text after processing all chars,
        # but only if there's no incomplete pattern being buffered. A trailing
        # partial start delimiter (e.g. "<thin" of "<think>") is held back so a
        # delimiter split across chunks isn't leaked as plain text; it's
        # retained in the buffer to be completed by the next chunk.
        if self._aggregation_type == AggregationType.TOKEN and self._text:
            if self._match_start_of_pattern(self._text) is None:
                held_back = longest_trailing_partial_match(
                    self._text, [pattern["start"] for pattern in self._patterns.values()]
                )
                yield_length = len(self._text) - held_back
                if yield_length > 0:
                    content = self._text[:yield_length]
                    self._text = self._text[yield_length:]
                    self._last_processed_position = len(self._text)
                    yield PatternMatch(
                        content=content,
                        type=AggregationType.TOKEN,
                        full_match=content,
                    )

    async def flush(self) -> Aggregation | None:
        """Flush any remaining text in the buffer.

        Degrades incomplete pattern state the same way a closed pair's
        action would, without knowing whether the pair would ever have
        closed:

        - REMOVE and AGGREGATE: an unclosed start delimiter and everything
          from it onward is dropped (their content is either stripped or
          treated as a side channel once closed, so it's never spoken
          incomplete). The pattern handler is not invoked. Text before the
          earliest such unclosed start delimiter is kept.
        - KEEP: a closed KEEP pair is spoken verbatim, delimiters included
          (see :meth:`aggregate`), so an unclosed one degrades to the same
          thing rather than being dropped.

        If the buffer ends with a proper prefix of a REMOVE/AGGREGATE start
        delimiter (the stream cut off mid-delimiter), that trailing partial
        text is trimmed too. This accepts an inherent ambiguity: a legitimate
        trailing character sequence that happens to match a delimiter prefix
        (e.g. a stray ``<``) is trimmed the same way.

        Returns:
            Any remaining text with incomplete REMOVE/AGGREGATE pattern
            content dropped, or None if there is nothing left to return.
        """
        cut_index = self._find_earliest_droppable_start(self._text)
        result_text = self._text if cut_index is None else self._text[:cut_index]
        result_text = self._trim_trailing_partial_start(result_text)

        await self.reset()

        stripped = result_text.strip()
        if not stripped:
            return None
        agg_type = (
            AggregationType.TOKEN
            if self._aggregation_type == AggregationType.TOKEN
            else AggregationType.SENTENCE
        )
        return PatternMatch(content=stripped, type=agg_type, full_match=result_text)

    def _find_earliest_droppable_start(self, text: str) -> int | None:
        """Find the earliest unmatched start delimiter that flush() should cut at.

        For each registered pattern, replicates the same left-to-right,
        non-overlapping pairing that :meth:`_process_complete_patterns` uses
        (earliest start paired with earliest following end) to determine
        which start-delimiter occurrences are still unmatched. Only
        REMOVE/AGGREGATE patterns' unmatched starts are candidates: KEEP
        patterns are excluded because unclosed KEEP content is kept, not cut.

        Args:
            text: The buffered text to scan.

        Returns:
            The index of the earliest droppable unmatched start delimiter,
            or None if there isn't one.
        """
        earliest = None

        for pattern_info in self._patterns.values():
            if pattern_info.get("action", MatchAction.REMOVE) == MatchAction.KEEP:
                continue

            start = pattern_info["start"]
            end = pattern_info["end"]
            regex = f"{re.escape(start)}(.*?){re.escape(end)}"
            matched_spans = [match.span() for match in re.finditer(regex, text, re.DOTALL)]

            search_from = 0
            while True:
                index = text.find(start, search_from)
                if index == -1:
                    break

                span = next((s for s in matched_spans if s[0] <= index < s[1]), None)
                if span is not None:
                    # This occurrence belongs to a completed match; skip past
                    # it entirely so a start nested inside its content isn't
                    # mistaken for an unmatched one.
                    search_from = span[1]
                    continue

                if earliest is None or index < earliest:
                    earliest = index
                break

        return earliest

    def _trim_trailing_partial_start(self, text: str) -> str:
        """Trim a trailing partial REMOVE/AGGREGATE start delimiter, if any.

        The stream can end mid-delimiter (e.g. buffered text ending in
        ``<te`` of ``<test>``). That partial text isn't a complete start
        delimiter, so :meth:`_find_earliest_droppable_start` won't catch it,
        but it also isn't safe to speak. KEEP patterns are excluded since a
        KEEP delimiter is kept whether or not it's complete.

        Args:
            text: The text to check for a trailing partial delimiter.

        Returns:
            The text with a trailing partial start delimiter removed, or the
            original text if there isn't one.
        """
        max_trim = 0

        for pattern_info in self._patterns.values():
            if pattern_info.get("action", MatchAction.REMOVE) == MatchAction.KEEP:
                continue

            start = pattern_info["start"]
            for length in range(min(len(start) - 1, len(text)), 0, -1):
                if text.endswith(start[:length]):
                    max_trim = max(max_trim, length)
                    break

        return text[:-max_trim] if max_trim else text

    async def handle_interruption(self):
        """Handle interruptions by clearing the buffer and pattern state.

        Called when an interruption occurs in the processing pipeline,
        to reset the state and discard any partially aggregated text.
        """
        await super().handle_interruption()
        self._last_processed_position = 0
        # Pattern and handler state persists across interruptions

    async def reset(self):
        """Clear the internally aggregated text.

        Resets the aggregator to its initial state, discarding any
        buffered text and clearing pattern tracking state.
        """
        await super().reset()
        self._last_processed_position = 0
        # Pattern and handler state persists across resets
