#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Word completion tracker for TTS context ordering."""

import re
import unicodedata

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

    def __init__(
        self,
        expected_text: str,
        raw_text: str | None = None,
    ):
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

        # _expected_raw_text is the original expected_text before normalization.
        # _expected_raw_pos is a cursor into it, advanced by the same alnum count
        # as the TTS word stream, so the force-complete path can emit the remaining
        # unspoken text as a TTSTextFrame instead of silently dropping it.
        self._expected_raw_text = expected_text
        self._expected_raw_pos = 0

        # raw_text is the original LLM-produced text (with pattern delimiters like
        # <card>...</card>). We track _raw_pos as a byte cursor into it, advancing
        # by the same number of alphanumeric chars consumed from the TTS word stream.
        self._raw_text = raw_text
        self._raw_pos = 0

        self._overflow: str | None = None
        self._raw_overflow_word: str | None = None
        self._raw_consumed: str | None = None
        self._frame_word: str | None = None
        self._includes_inter_frame_spaces = False
        logger.debug(f"WordCompletionTracker: {self._expected}")

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip XML/HTML tags then keep only lowercase alphanumeric characters."""
        text = re.sub(r"<[^>]+>", "", text)
        # Decompose accents (ā → a + ̄)
        text = unicodedata.normalize("NFKD", text)
        # Keep base characters only
        text = "".join(c for c in text if c.isalnum())
        return text.lower()

    @staticmethod
    def _advance_by_alnums(text: str, start_pos: int, n: int) -> int:
        """Return the position in *text* after advancing past *n* alphanumeric chars.

        Moves through the text one character at a time, counting only alphanumeric
        characters. XML/HTML tags (``<...>``) are skipped entirely — their content
        is not counted against the budget, so the returned span includes the full tag.
        Other non-alphanumeric characters (spaces, punctuation) are also passed over
        without decrementing the budget.
        """
        pos = start_pos
        count = 0
        while pos < len(text) and count < n:
            if text[pos] == "<":
                end = text.find(">", pos)
                pos = end + 1 if end != -1 else pos + 1
            elif text[pos].isalnum():
                count += 1
                pos += 1
            else:
                pos += 1

        # NEW: consume trailing punctuation (but not tags)
        while pos < len(text):
            if text[pos] == "<":
                break
            if text[pos].isalnum() or text[pos].isspace():
                break
            pos += 1

        return pos

    def add_word_and_check_complete(
        self, word: str, includes_inter_frame_spaces: bool | None = False
    ) -> bool:
        """Record a spoken word from a word-timestamp event.

        Normalizes ``word``, appends it to the running total, and checks whether
        all expected alphanumeric characters have been covered.

        Before advancing, checks whether the word belongs to this frame via
        ``word_belongs_here``. If it does not (e.g. the TTS provider silently
        dropped a word-timestamp), the slot is force-completed: the remaining
        unspoken text from ``expected_text`` is stored in ``_frame_word`` so a
        TTSTextFrame can still be emitted for the dropped portion, all remaining
        ``raw_text`` is consumed, and the entire incoming word is set as overflow
        so the caller's overflow path routes it to the next slot unchanged.

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
            includes_inter_frame_spaces: Whether the word includes inter-frame spaces.

        Returns:
            True when all expected content has been covered.
        """
        normalized = self._normalize(word)
        self._includes_inter_frame_spaces = includes_inter_frame_spaces

        prev_len = len(self._received)
        expected_len = len(self._expected)

        self._overflow = None
        self._raw_overflow_word = None
        self._raw_consumed = None
        self._frame_word = None

        if prev_len > expected_len:
            logger.warning(f"{self}, trying to add a word in an already complete frame")
            return True

        # If the word doesn't match the next expected chars, the TTS provider
        # likely dropped a word-timestamp event. Force-complete this slot: emit the
        # remaining expected text as _frame_word so a TTSTextFrame is still produced
        # for the unspoken portion, consume all remaining raw_text, and route the
        # entire incoming word as overflow for the next slot.
        if not self.word_belongs_here(word):
            self._frame_word = self._expected_raw_text[self._expected_raw_pos :]
            if self._raw_text is not None:
                self._raw_consumed = self._raw_text[self._raw_pos :]
                self._raw_pos = len(self._raw_text)
                # This should not happen: force-complete sweeps all remaining
                # raw_text, so the span must contain the frame word. If it
                # doesn't, expected_text and raw_text are out of sync in an
                # unexpected way — discard rather than returning a corrupt span.
                if self._frame_word and self._frame_word not in self._raw_consumed:
                    logger.warning(
                        f"WordCompletionTracker: force-complete raw_consumed {repr(self._raw_consumed)!s} "
                        f"does not contain frame_word {repr(self._frame_word)!s}, discarding"
                    )
                    self._raw_consumed = None
            self._received = self._expected  # force-complete
            self._overflow = normalized
            self._raw_overflow_word = word
            return True

        self._received += normalized

        # How many normalized chars from this word belong to the current frame.
        chars_for_frame = min(len(normalized), expected_len - prev_len)

        if prev_len + len(normalized) > expected_len:
            # This word straddles the frame boundary. Split into:
            #   - _frame_word: the prefix of `word` up to the split point, used
            #     for the TTSTextFrame of the current slot.
            #   - normalized overflow: fed into the next frame's tracker.
            #   - raw overflow word: the raw suffix after the split point, used
            #     to build a TTSTextFrame attributed to the next AggregatedTextFrame.
            split_pos = self._advance_by_alnums(word, 0, chars_for_frame)
            self._frame_word = word[:split_pos]
            self._overflow = normalized[expected_len - prev_len :]
            self._raw_overflow_word = word[split_pos:]
        else:
            # Word fits entirely in this frame.
            self._frame_word = word

        # Advance the expected-raw cursor by the same alnum count so the
        # force-complete path knows where in _expected_raw_text to start from.
        self._expected_raw_pos = self._advance_by_alnums(
            self._expected_raw_text, self._expected_raw_pos, chars_for_frame
        )

        if self._raw_text is not None:
            if self.is_complete:
                # Consume all remaining raw text so that closing tags (e.g.
                # </card>) are included in this frame's TTSTextFrame rather
                # than silently dropped.
                self._raw_consumed = self._raw_text[self._raw_pos :]
                self._raw_pos = len(self._raw_text)
            else:
                if chars_for_frame == 0:
                    # Consume exactly the raw word (including preceding spaces if present)
                    start = self._raw_pos
                    if not self._includes_inter_frame_spaces:
                        # Skip leading spaces (they belong to previous token)
                        while start < len(self._raw_text) and self._raw_text[start].isspace():
                            start += 1
                    end = start + len(word)
                    self._raw_consumed = self._raw_text[start:end]
                    self._raw_pos = end
                else:
                    # Advance through raw_text by exactly chars_for_frame alphanumeric
                    # chars. Non-alnum chars (spaces, opening tags) are included in the
                    # slice, preserving the original formatting for the context.
                    new_pos = self._advance_by_alnums(
                        self._raw_text, self._raw_pos, chars_for_frame
                    )
                    self._raw_consumed = self._raw_text[self._raw_pos : new_pos]
                    self._raw_pos = new_pos
            # This should not happen: the raw cursor is driven by the same
            # alnum count as the word stream, so the consumed span must contain
            # the frame word. If it doesn't, the cursors drifted out of sync
            # in an unexpected way — discard rather than returning a corrupt span.
            if self._frame_word and self._frame_word not in self._raw_consumed:
                logger.warning(
                    f"WordCompletionTracker: raw_consumed {repr(self._raw_consumed)!s} "
                    f"does not contain frame_word {repr(self._frame_word)!s}, discarding"
                )
                self._raw_consumed = None

        return self.is_complete

    def word_belongs_here(self, word: str) -> bool:
        """Return True if this word plausibly belongs to the remaining expected text.

        Checks whether the normalized word is a prefix match for the remaining
        expected chars. Used to detect when the TTS provider silently dropped a
        word-timestamp event: if the incoming word does not match the next expected
        chars of the current slot, the caller should skip to the next slot rather
        than advancing this tracker with wrong content.

        For example, if the remaining expected is ``"number"`` and the word is
        ``"4111"``, the check fails → skip to the next slot. If remaining is
        ``"4111111111111111"`` and word is ``"4111"``, the check passes → correct slot.

        Punctuation/whitespace-only words (empty after normalization) are neutral
        and always return True.
        """
        normalized = self._normalize(word)
        if not normalized:
            return True
        remaining = self._expected[len(self._received) :]
        if not remaining:
            return False
        # The word belongs here if its normalized chars are a prefix of the
        # remaining expected text (handles both full words and partial tokens).
        check_len = min(len(normalized), len(remaining))
        return remaining.startswith(normalized[:check_len])

    def get_frame_word(self) -> str | None:
        """Return the portion of the last word that belongs to this frame.

        - Normal word (no overflow): the full word.
        - Straddling word: the prefix up to the frame boundary (e.g. ``"1111"``
          from ``"1111 And"``).
        - Force-completed (word didn't belong): the remaining unspoken text from
          ``expected_text`` so a TTSTextFrame can still be emitted for the dropped
          portion. The incoming word is routed as overflow to the next slot.
        """
        return (
            self._frame_word.strip()
            if self._frame_word and not self._includes_inter_frame_spaces
            else self._frame_word
        )

    def get_overflow(self) -> str | None:
        """Return normalized overflow from the last added word, if any."""
        return (
            self._overflow.strip()
            if self._overflow and not self._includes_inter_frame_spaces
            else self._overflow
        )

    def get_raw_overflow_word(self) -> str | None:
        """Return the raw suffix of the last word that overflows into the next frame.

        Unlike ``get_overflow()`` (which is normalized), this preserves the original
        casing and any non-alphanumeric characters so the overflow TTSTextFrame has
        natural word text.
        """
        return (
            self._raw_overflow_word.strip()
            if self._raw_overflow_word and not self._includes_inter_frame_spaces
            else self._raw_overflow_word
        )

    def get_raw_consumed(self) -> str | None:
        """Return the raw text span consumed from raw_text for the last added word.

        Returns None if no raw_text was provided at construction time.
        """
        return (
            self._raw_consumed.strip()
            if self._raw_consumed and not self._includes_inter_frame_spaces
            else self._raw_consumed
        )

    def get_remaining_text(self) -> str:
        """Return the unspoken portion of expected_text, stripped of leading/trailing whitespace.

        This is the text that the TTS provider has not yet confirmed via word-timestamp
        events. Useful for force-completing a slot when the audio context ends before all
        word-timestamp events have arrived.
        """
        return self._expected_raw_text[self._expected_raw_pos :].strip()

    def get_remaining_raw_text(self) -> str | None:
        """Return the unspoken portion of raw_text, stripped of leading/trailing whitespace.

        Returns None if no raw_text was provided at construction time. Like
        ``get_remaining_text()``, intended for force-completing a slot so that the
        conversation context receives the full original text.
        """
        if self._raw_text is None:
            return None
        remaining = self._raw_text[self._raw_pos :].strip()
        return remaining if remaining else None

    @property
    def is_complete(self) -> bool:
        """True when accumulated normalized chars >= expected normalized chars."""
        return len(self._received) >= len(self._expected)

    def reset(self):
        """Reset received word accumulation without changing the expected text."""
        self._received = ""
        self._expected_raw_pos = 0
        self._raw_pos = 0
        self._overflow = None
        self._raw_overflow_word = None
        self._raw_consumed = None
        self._frame_word = None
