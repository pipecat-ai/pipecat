#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simple text aggregator for basic sentence-boundary text processing.

This module provides a straightforward text aggregator that accumulates text
until it finds an end-of-sentence marker, making it suitable for basic TTS
text processing scenarios.
"""

from collections.abc import AsyncIterator
from enum import Enum, auto

from pipecat.utils.string import SENTENCE_ENDING_PUNCTUATION, match_endofsentence
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType, BaseTextAggregator


class _LookaheadState(Enum):
    """Track lookahead across chunks for early emission and a word-end retry.

    Clear boundaries can emit at the first character; unresolved ones wait for
    a complete word. For example, "Hello. N" can emit without waiting for "Next ".
    Waiting for word completion added up to 297 ms in our Anthropic sample.

    Unresolved boundaries retry only when the following word ends. In “Albert I.
    Douglas”, checking the partial word “Do” could mistake it for a sentence
    starter and split after the initial.
    """

    IDLE = auto()
    AWAITING_CHARACTER = auto()
    AWAITING_WORD = auto()
    IN_WORD = auto()


class SimpleTextAggregator(BaseTextAggregator):
    """Simple text aggregator that accumulates text until sentence boundaries.

    This aggregator provides basic functionality for accumulating text tokens
    and releasing them when an end-of-sentence marker is detected. It's the
    most straightforward implementation of text aggregation for TTS processing.
    """

    def __init__(self, **kwargs):
        """Initialize the simple text aggregator.

        Creates an empty text buffer ready to begin accumulating text tokens.

        Args:
            **kwargs: Additional arguments passed to BaseTextAggregator (e.g. aggregation_type).
        """
        super().__init__(**kwargs)
        self._text = ""
        self._lookahead_state = _LookaheadState.IDLE

    @property
    def text(self) -> Aggregation:
        """Get the currently aggregated text.

        Returns:
            The text that has been accumulated in the buffer.
        """
        return Aggregation(text=self._text.strip(" "), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Aggregate text and yield completed aggregations.

        In SENTENCE mode, processes the input text character-by-character. When
        sentence-ending punctuation is detected, it waits for non-whitespace
        lookahead before calling sentencex.

        In TOKEN mode, yields the text immediately without buffering.

        Args:
            text: Text to aggregate.

        Yields:
            Aggregation objects (sentences in SENTENCE mode, tokens in TOKEN mode).
        """
        if self._aggregation_type == AggregationType.TOKEN:
            if text:
                yield Aggregation(text=text, type=AggregationType.TOKEN)
            return

        # Process text character by character
        for char in text:
            self._text += char

            # Check for sentence with lookahead
            result = await self._check_sentence_with_lookahead(char)
            if result:
                yield result

    async def _check_sentence_with_lookahead(self, char: str) -> Aggregation | None:
        """Check for sentence boundaries using lookahead logic.

        This method implements the core sentence detection logic with lookahead.
        When sentence-ending punctuation is detected, it waits for the next
        non-whitespace character before calling sentencex. This disambiguates cases
        like "$29." (not a sentence) vs "$29. Next" (sentence ends at period).
        Whitespace alone is not meaningful lookahead since it appears in both
        cases. Instead, the first non-whitespace character after the punctuation
        is used to confirm the sentence boundary. An unresolved candidate is
        checked again when the following word ends.

        Subclasses can call this via super() to reuse the lookahead behavior
        while adding their own logic (e.g., tag handling, pattern matching).

        Args:
            char: The most recently added character (used for lookahead check).

        Returns:
            Aggregation if sentence found, None otherwise.
        """
        is_punctuation = char in SENTENCE_ENDING_PUNCTUATION
        result = None

        if self._advance_lookahead(char, is_punctuation=is_punctuation):
            # New punctuation can finish the lookahead word, but must not make
            # the tokenizer's terminal-punctuation fallback accept the whole buffer.
            candidate = self._text[:-1] if is_punctuation else self._text
            eos_marker = match_endofsentence(candidate, language=self.language)
            if eos_marker:
                result = Aggregation(
                    text=self._text[:eos_marker].strip(" "), type=AggregationType.SENTENCE
                )
                self._text = self._text[eos_marker:]
                self._lookahead_state = _LookaheadState.IDLE

        # Punctuation starts a fresh candidate, even if it also completed a retry.
        if is_punctuation:
            self._lookahead_state = _LookaheadState.AWAITING_CHARACTER

        return result

    def _advance_lookahead(self, char: str, *, is_punctuation: bool) -> bool:
        """Advance the pending candidate and decide whether to check its boundary.

        Check once at the first non-whitespace character, then once more when
        the following word ends. Opening quotes or symbols do not start a word.

        If the first tokenizer check finds no boundary::

            'I. '      -> AWAITING_CHARACTER
            'I. "'     -> AWAITING_WORD (check at the quote)
            'I. "D'    -> IN_WORD (no check)
            'I. "Did ' -> IDLE (retry at the word-ending space)

        The caller performs the tokenizer check when this method returns True.

        Args:
            char: The most recently appended character.
            is_punctuation: Whether the character is sentence-ending punctuation.

        Returns:
            Whether the buffered text is ready for a tokenizer check.
        """
        match self._lookahead_state:
            case _LookaheadState.IDLE:
                return False

            case _LookaheadState.AWAITING_CHARACTER:
                if char.isspace() or is_punctuation:
                    return False
                self._lookahead_state = (
                    _LookaheadState.IN_WORD if char.isalnum() else _LookaheadState.AWAITING_WORD
                )
                # Ordinary boundaries such as "Hello. N" can emit immediately.
                return True

            case _LookaheadState.AWAITING_WORD:
                if char.isalnum():
                    self._lookahead_state = _LookaheadState.IN_WORD
                return False

            case _LookaheadState.IN_WORD:
                # Never retry at a partial prefix such as "Do" inside "Douglas".
                if char.isspace() or is_punctuation:
                    self._lookahead_state = _LookaheadState.IDLE
                    return True
                return False

    async def flush(self) -> Aggregation | None:
        """Flush any remaining text in the buffer.

        Returns any text remaining in the buffer. This is called at the end
        of a stream to ensure all text is processed. In TOKEN mode, returns
        None since tokens are yielded immediately.

        Returns:
            Any remaining text as a sentence, or None if buffer is empty or in TOKEN mode.
        """
        if self._aggregation_type == AggregationType.TOKEN:
            return None

        if self._text:
            # Return whatever we have in the buffer
            result = self._text
            await self.reset()
            return Aggregation(text=result.strip(" "), type=AggregationType.SENTENCE)
        return None

    async def handle_interruption(self):
        """Handle interruptions by clearing the text buffer.

        Called when an interruption occurs in the processing pipeline,
        discarding any partially accumulated text.
        """
        self._text = ""
        self._lookahead_state = _LookaheadState.IDLE

    async def reset(self):
        """Clear the internally aggregated text.

        Resets the aggregator to its initial empty state, discarding
        any accumulated text content.
        """
        self._text = ""
        self._lookahead_state = _LookaheadState.IDLE
