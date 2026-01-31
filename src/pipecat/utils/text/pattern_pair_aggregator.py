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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

from loguru import logger

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class MatchAction(Enum):
    """Actions to take when a pattern pair is matched.

    Parameters:
        REMOVE: The text along with its delimiters will be removed from the streaming text.
            Sentence aggregation will continue on as if this text did not exist.
        KEEP: The matched pattern will be kept in the text (including delimiters).
            Sentence aggregation will continue on with the pattern included as normal text.
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


@dataclass(frozen=True)
class _PatternSpec:
    type: str
    start: str
    end: str
    action: MatchAction

    @property
    def start_len(self) -> int:
        return len(self.start)

    @property
    def end_len(self) -> int:
        return len(self.end)


@dataclass
class _OpenPattern:
    spec: _PatternSpec
    start_idx: int


class PatternPairAggregator(SimpleTextAggregator):
    """Aggregator that identifies and processes content between pattern pairs.

    This aggregator buffers text until it can identify complete pattern pairs
    (defined by start and end patterns), processes the content between these
    patterns using registered handlers. By default, its aggregation method
    returns text at sentence boundaries, and remove the content found between
    any matched patterns. However,    matched patterns can also be configured to be returned as a separate aggregation
    object containing the content between their start and end patterns, removed entirely,
    or kept in-stream (including delimiters) while still triggering a handler.

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
        super().__init__()
        self._patterns: Dict[str, dict] = {}
        self._specs: Dict[str, _PatternSpec] = {}
        self._handlers: Dict[str, Callable[[PatternMatch], Awaitable[None]]] = {}

        self._open: List[_OpenPattern] = []

        self._start_by_last: Dict[str, List[_PatternSpec]] = {}
        self._end_by_last: Dict[str, List[_PatternSpec]] = {}

    @property
    def text(self) -> Aggregation:
        """Get the currently aggregated text.

        Returns:
            The text that has been accumulated in the buffer.
        """
        stripped_text = self._text.strip()
        if self._open:
            return Aggregation(text=stripped_text, type=self._open[-1].spec.type)
        return Aggregation(text=stripped_text, type=AggregationType.SENTENCE)

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
        if type in [AggregationType.SENTENCE, AggregationType.WORD]:
            raise ValueError(
                f"The aggregation type '{type}' is reserved for default behavior and can not be used for custom patterns."
            )

        if not start_pattern or not end_pattern:
            raise ValueError("start_pattern and end_pattern must be non-empty strings")

        old = self._specs.get(type)
        if old is not None:
            try:
                self._start_by_last.get(old.start[-1], []).remove(old)
            except ValueError:
                pass
            try:
                self._end_by_last.get(old.end[-1], []).remove(old)
            except ValueError:
                pass

        spec = _PatternSpec(type=type, start=start_pattern, end=end_pattern, action=action)

        self._patterns[type] = {
            "start": start_pattern,
            "end": end_pattern,
            "type": type,
            "action": action,
        }
        self._specs[type] = spec

        self._start_by_last.setdefault(start_pattern[-1], []).append(spec)
        self._end_by_last.setdefault(end_pattern[-1], []).append(spec)

        self._start_by_last[start_pattern[-1]].sort(key=lambda s: s.start_len, reverse=True)
        self._end_by_last[end_pattern[-1]].sort(key=lambda s: s.end_len, reverse=True)

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

    def _push_open_if_start_delimiter(self) -> Optional[_OpenPattern]:
        if not self._text:
            return None
        last = self._text[-1]
        for spec in self._start_by_last.get(last, []):
            if len(self._text) >= spec.start_len and self._text.endswith(spec.start):
                start_idx = len(self._text) - spec.start_len
                op = _OpenPattern(spec=spec, start_idx=start_idx)
                self._open.append(op)
                return op
        return None

    async def _close_one_if_end_delimiter(self) -> Optional[Tuple[_PatternSpec, PatternMatch]]:
        if not self._text or not self._open:
            return None

        last = self._text[-1]
        for spec in self._end_by_last.get(last, []):
            if len(self._text) < spec.end_len or not self._text.endswith(spec.end):
                continue

            open_idx = None
            for i in range(len(self._open) - 1, -1, -1):
                if self._open[i].spec.type == spec.type:
                    open_idx = i
                    break
            if open_idx is None:
                continue

            start_idx = self._open[open_idx].start_idx
            if start_idx < 0 or start_idx > len(self._text) - spec.end_len:
                continue
            if not self._text.startswith(spec.start, start_idx):
                continue

            end_idx = len(self._text)
            full_match = self._text[start_idx:end_idx]
            content = full_match[spec.start_len : -spec.end_len]
            pm = PatternMatch(content=content.strip(), type=spec.type, full_match=full_match)

            del self._open[open_idx:]

            handler = self._handlers.get(spec.type)
            if handler is not None:
                try:
                    await handler(pm)
                except Exception as e:
                    logger.error(f"Error in pattern handler for {spec.type}: {e}")

            if spec.action in (MatchAction.REMOVE, MatchAction.AGGREGATE):
                self._text = self._text[:start_idx]

            return spec, pm

        return None

    async def aggregate(self, text: str) -> AsyncIterator[PatternMatch]:
        """Aggregate text and process pattern pairs.

        Processes the input text character-by-character, handles pattern pairs,
        and uses the parent's lookahead logic for sentence detection when no
        patterns are active.

        Args:
            text: Text to aggregate.

        Yields:
            PatternMatch objects as patterns complete or sentences are detected.
        """
        if not self._patterns and not self._open:
            async for aggr in super().aggregate(text):
                yield PatternMatch(content=aggr.text, type=aggr.type, full_match=aggr.text)
            return

        for char in text:
            self._text += char
            yielded_aggregate = False

            while True:
                closed = await self._close_one_if_end_delimiter()
                if closed is None:
                    break
                spec, pm = closed
                if spec.action == MatchAction.AGGREGATE:
                    yield pm
                    yielded_aggregate = True
                    break

            if yielded_aggregate:
                continue

            was_open = bool(self._open)
            opened = self._push_open_if_start_delimiter()

            if (
                opened is not None
                and not was_open
                and opened.spec.action == MatchAction.AGGREGATE
                and opened.start_idx > 0
            ):
                prefix = self._text[: opened.start_idx]
                self._text = self._text[opened.start_idx :]
                opened.start_idx = 0
                yield PatternMatch(content=prefix.strip(), type=AggregationType.SENTENCE, full_match=prefix)
                continue

            if self._open:
                continue

            aggregation = await super()._check_sentence_with_lookahead(char)
            if aggregation:
                yield PatternMatch(
                    content=aggregation.text, type=aggregation.type, full_match=aggregation.text
                )

    async def handle_interruption(self):
        """Handle interruptions by clearing the buffer and pattern state.

        Called when an interruption occurs in the processing pipeline,
        to reset the state and discard any partially aggregated text.
        """
        await super().handle_interruption()
        self._open.clear()

    async def reset(self):
        """Clear the internally aggregated text.

        Resets the aggregator to its initial state, discarding any
        buffered text and clearing pattern tracking state.
        """
        await super().reset()
        self._open.clear()
