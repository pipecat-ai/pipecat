#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Word completion tracker for TTS context ordering."""

import re
from typing import Optional

from loguru import logger


class WordCompletionTracker:
    """Tracks whether all words from a source AggregatedTextFrame have been spoken.

    Compares normalized alphanumeric character counts between the expected text
    and accumulated spoken words, making the check robust to punctuation, spacing,
    and XML/HTML tags (e.g. SSML tags like ``<spell>...</spell>`` returned by some
    TTS providers in word-timestamp events).

    When ``raw_text`` is provided (e.g. the original pattern-matched text including
    delimiters like ``<card>4111 1111 1111 1111</card>``), the tracker additionally
    maps each spoken word back to its corresponding span in that raw text. This
    lets callers attach the original text to ``TTSTextFrame`` entries so the
    conversation context receives properly-tagged content rather than the cleaned
    words received from the TTS provider.

    Background: TTS providers apply their own SSML tags to the text before
    synthesis and return word-timestamp events containing the raw spoken words
    (e.g. ``"4111"``, ``"1111"``). Without raw-text tracking, the conversation
    context would only see those cleaned words and lose the original structure
    (e.g. ``<card>4111 1111 1111 1111</card>``). By mapping normalized char counts
    back to positions in ``raw_text``, each TTSTextFrame can carry the exact span
    of original text it represents.

    Overflow handling: TTS providers sometimes return a single word token that
    spans the boundary between two AggregatedTextFrames (e.g. ``"1111</spell>And"``
    when one frame ends with ``1111</card>`` and the next begins with ``And``). The
    tracker detects this, exposes the normalized overflow via ``get_overflow()``
    and the raw word suffix via ``get_raw_overflow_word()``, so callers can feed the
    remainder into the next frame's tracker and emit a correctly-attributed
    TTSTextFrame for each part.

    Example::

        tracker = WordCompletionTracker("Hello, world!")
        tracker.add_word_and_check_complete("Hello")   # False
        tracker.add_word_and_check_complete("world")   # True  — normalized "helloworld" >= "helloworld"
    """

    def __init__(self, expected_text: str, raw_text: str | None = None):
        """Initialize the tracker with the text of the frame being spoken.

        Args:
            expected_text: Full text of the AggregatedTextFrame sent to TTS
                (may include TTS-specific SSML tags). Used only for normalized
                char-count completion tracking.
            raw_text: Original pattern-matched text including delimiters (e.g.
                ``<card>4111 1111 1111 1111</card>``). When provided, each
                ``add_word_and_check_complete`` call also returns the corresponding
                raw span via ``get_raw_consumed()``. Both texts normalize to the
                same alphanumeric sequence, so the same char-count cursor drives
                position tracking in both.
        """
        self._expected = self._normalize(expected_text)
        self._received = ""

        # raw_text is the original LLM-produced text (with pattern delimiters like
        # <card>...</card>). We track _raw_pos as a byte cursor into it, advancing
        # by the same number of alphanumeric chars consumed from the TTS word stream.
        self._raw_text = raw_text
        self._raw_pos = 0

        self._overflow: str | None = None
        self._raw_overflow_word: str | None = None
        self._raw_consumed: str | None = None
        logger.info(f"WordCompletionTracker: {self._expected}")

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip XML/HTML tags then keep only lowercase alphanumeric characters."""
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @staticmethod
    def _advance_by_alnums(text: str, start_pos: int, n: int) -> int:
        """Return the position in *text* after advancing past *n* alphanumeric chars.

        Moves through the text one character at a time, counting only alphanumeric
        characters. Non-alphanumeric characters (spaces, punctuation, tags) are
        passed over without decrementing the budget, so they end up included in
        the returned span.
        """
        pos = start_pos
        count = 0
        while pos < len(text) and count < n:
            if text[pos].isalnum():
                count += 1
            pos += 1
        return pos

    def add_word_and_check_complete(self, word: str) -> bool:
        """Record a spoken word from a word-timestamp event.

        Normalizes ``word``, appends it to the running total, and checks whether
        all expected alphanumeric characters have been covered.

        If ``raw_text`` was provided at construction time, also advances the raw
        cursor by the same number of alphanumeric chars consumed from this word and
        stores the corresponding raw span in ``_raw_consumed``. When this word
        completes the frame, the entire remaining raw text (including any closing
        tags) is consumed so nothing is lost.

        If the word overshoots the expected length (overflow), the normalized excess
        is stored in ``_overflow`` and the raw suffix of the word (everything after
        the last char belonging to this frame) is stored in ``_raw_overflow_word``,
        so the caller can attribute them to the next AggregatedTextFrame.

        Args:
            word: A single word token returned by the TTS service.

        Returns:
            True when all expected content has been covered.
        """
        normalized = self._normalize(word)

        prev_len = len(self._received)
        expected_len = len(self._expected)

        self._received += normalized
        self._overflow = None
        self._raw_overflow_word = None
        self._raw_consumed = None

        if prev_len >= expected_len:
            logger.warning(f"{self}, trying to add a word in a already complete frame")
            return True

        # How many normalized chars from this word belong to the current frame.
        chars_for_frame = min(len(normalized), expected_len - prev_len)

        if prev_len + len(normalized) > expected_len:
            # This word straddles the frame boundary. Split into:
            #   - normalized overflow: fed into the next frame's tracker
            #   - raw overflow word: the raw suffix of `word` starting after the
            #     chars_for_frame-th alphanumeric character; used to build a
            #     TTSTextFrame attributed to the next AggregatedTextFrame.
            self._overflow = normalized[expected_len - prev_len :]
            split_pos = self._advance_by_alnums(word, 0, chars_for_frame)
            self._raw_overflow_word = word[split_pos:]

        if self._raw_text is not None:
            if self.is_complete:
                # Consume all remaining raw text so that closing tags (e.g.
                # </card>) are included in this frame's TTSTextFrame rather
                # than silently dropped.
                self._raw_consumed = self._raw_text[self._raw_pos :]
                self._raw_pos = len(self._raw_text)
            else:
                # Advance through raw_text by exactly chars_for_frame alphanumeric
                # chars. Non-alnum chars (spaces, opening tags) are included in the
                # slice, preserving the original formatting for the context.
                new_pos = self._advance_by_alnums(self._raw_text, self._raw_pos, chars_for_frame)
                self._raw_consumed = self._raw_text[self._raw_pos : new_pos]
                self._raw_pos = new_pos

        return self.is_complete

    def get_overflow(self) -> str | None:
        """Return normalized overflow from the last added word, if any."""
        return self._overflow

    def get_raw_overflow_word(self) -> str | None:
        """Return the raw suffix of the last word that overflows into the next frame.

        Unlike ``get_overflow()`` (which is normalized), this preserves the original
        casing and any non-alphanumeric characters so the overflow TTSTextFrame has
        natural word text.
        """
        return self._raw_overflow_word

    def get_raw_consumed(self) -> str | None:
        """Return the raw text span consumed from raw_text for the last added word.

        Returns None if no raw_text was provided at construction time.
        """
        return self._raw_consumed

    @property
    def is_complete(self) -> bool:
        """True when accumulated normalized chars >= expected normalized chars."""
        return len(self._received) >= len(self._expected)

    def reset(self):
        """Reset received word accumulation without changing the expected text."""
        self._received = ""
        self._raw_pos = 0
        self._overflow = None
        self._raw_overflow_word = None
        self._raw_consumed = None
