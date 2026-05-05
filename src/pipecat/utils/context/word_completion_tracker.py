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

    Compares normalized alphanumeric character counts between the TTS text and
    accumulated spoken words, making the check robust to punctuation, spacing,
    and XML/HTML tags (e.g. SSML tags like ``<spell>...</spell>`` returned by some
    TTS providers in word-timestamp events).

    When ``llm_text`` is provided (e.g. the original pattern-matched text including
    delimiters like ``<card>4111 1111 1111 1111</card>``), the tracker additionally
    maps each spoken word back to its corresponding span in that LLM text. This
    lets callers attach the original text to ``TTSTextFrame`` entries so the
    conversation context receives properly-tagged content rather than the cleaned
    words received from the TTS provider.

    Background: TTS providers apply their own SSML tags to the text before
    synthesis and return word-timestamp events containing the raw spoken words
    (e.g. ``"4111"``, ``"1111"``). Without LLM-text tracking, the conversation
    context would only see those cleaned words and lose the original structure
    (e.g. ``<card>4111 1111 1111 1111</card>``). By mapping normalized char counts
    back to positions in ``llm_text``, each TTSTextFrame can carry the exact span
    of original text it represents.

    Overflow handling: TTS providers sometimes return a single word token that
    spans the boundary between two AggregatedTextFrames (e.g. ``"1111</spell>And"``
    when one frame ends with ``1111</card>`` and the next begins with ``And``). The
    tracker detects this and exposes the raw overflow suffix via ``get_overflow_word()``,
    so callers can feed the remainder into the next frame's tracker and emit a
    correctly-attributed TTSTextFrame for each part.

    Example::

        tracker = WordCompletionTracker("Hello, world!")
        tracker.add_word_and_check_complete("Hello")   # False
        tracker.add_word_and_check_complete("world")   # True  — normalized "helloworld" >= "helloworld"
    """

    def __init__(
        self,
        tts_text: str,
        llm_text: str | None = None,
    ):
        """Initialize the tracker with the text of the frame being spoken.

        Args:
            tts_text: Full text of the AggregatedTextFrame sent to TTS (may include
                TTS-specific SSML tags). Used for normalized char-count completion
                tracking and as the cursor reference for the TTS word stream.
            llm_text: Original LLM-produced text including pattern delimiters (e.g.
                ``<card>4111 1111 1111 1111</card>``). When provided, each
                ``add_word_and_check_complete`` call also returns the corresponding
                LLM span via ``get_llm_consumed()``. Both texts normalize to the
                same alphanumeric sequence, so the same char-count cursor drives
                position tracking in both.
        """
        self._tts_normalized = self._normalize(tts_text)
        self._received = ""

        # _tts_text is the original tts_text before normalization.
        # _tts_pos is a cursor into it, advanced by the same alnum count
        # as the TTS word stream, so the force-complete path can emit the remaining
        # unspoken text as a TTSTextFrame instead of silently dropping it.
        self._tts_text = tts_text
        self._tts_pos = 0

        # _llm_text is the original LLM-produced text (with pattern delimiters like
        # <card>...</card>). We track _llm_pos as a cursor into it, advancing
        # by the same number of alphanumeric chars consumed from the TTS word stream.
        self._llm_text = llm_text
        self._llm_pos = 0

        self._overflow_word: str | None = None
        self._llm_consumed: str | None = None
        self._frame_word: str | None = None
        logger.debug(f"WordCompletionTracker: {self._tts_normalized}")

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip XML/HTML tags then keep only lowercase alphanumeric characters."""
        text = re.sub(r"<[^>]+>", "", text)
        # Decompose accents (ā → a + ̄)
        text = unicodedata.normalize("NFKD", text)
        # Keep base characters only
        text = "".join(c for c in text if c.isalnum())
        return text.lower()

    # Typographic variants that LLMs commonly emit but TTS services normalize away.
    _TYPOGRAPHY_FOLD = str.maketrans(
        {
            "‘": "'",  # ' LEFT SINGLE QUOTATION MARK
            "’": "'",  # ' RIGHT SINGLE QUOTATION MARK
            "ʼ": "'",  # ʼ MODIFIER LETTER APOSTROPHE
            "“": '"',  # " LEFT DOUBLE QUOTATION MARK
            "”": '"',  # " RIGHT DOUBLE QUOTATION MARK
            "–": "-",  # – EN DASH
            "—": "-",  # — EM DASH
        }
    )

    @staticmethod
    def _fold_typography(text: str) -> str:
        """Replace typographic punctuation variants with their ASCII equivalents."""
        return text.translate(WordCompletionTracker._TYPOGRAPHY_FOLD)

    @staticmethod
    def _remove_trailing_punctuation(text: str) -> str:
        """Remove punctuation only at the very end of the given text."""
        i = len(text)
        while i > 0 and unicodedata.category(text[i - 1]).startswith("P"):
            i -= 1
        return text[:i]

    @staticmethod
    def _advance_by_alnums(text: str, start_pos: int, n: int) -> int:
        """Return the position in *text* after advancing past *n* alphanumeric chars.

        Moves through the text one character at a time, counting only alphanumeric
        characters. XML/HTML tags (``<...>``) are skipped entirely — their content
        is not counted against the budget, so the returned span includes the full tag.
        Other non-alphanumeric characters (spaces, punctuation) are also passed over
        without decrementing the budget.

        After the *n* alnum chars are consumed, advances further past any immediately
        following punctuation (e.g. the ``,`` in ``"questions,"`` or the ``.`` in
        ``"done."``), stopping before the next space, alnum char, or XML tag.

        Args:
            text: The source text to scan.
            start_pos: Starting position in *text*.
            n: Number of alphanumeric characters to consume.
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

        while pos < len(text):
            if text[pos] == "<":
                break
            if text[pos].isalnum() or text[pos].isspace():
                break
            pos += 1

        return pos

    def add_word_and_check_complete(self, word: str) -> bool:
        """Record a spoken word from a word-timestamp event.

        Normalizes ``word``, appends it to the running total, and checks whether
        all expected alphanumeric characters have been covered.

        Before advancing, checks whether the word belongs to this frame via
        ``word_belongs_here``. If it does not (e.g. the TTS provider silently
        dropped a word-timestamp), the slot is force-completed: the remaining
        unspoken text from ``tts_text`` is stored in ``_frame_word`` so a
        TTSTextFrame can still be emitted for the dropped portion, all remaining
        ``llm_text`` is consumed, and the entire incoming word is set as overflow
        so the caller's overflow path routes it to the next slot unchanged.

        If ``llm_text`` was provided at construction time, also advances the LLM
        cursor by the same number of alphanumeric chars consumed from this word and
        stores the corresponding LLM span in ``_llm_consumed``. When this word
        completes the frame, the entire remaining LLM text (including any closing
        tags) is consumed so nothing is lost.

        If the word overshoots the expected length (overflow), the raw suffix of
        the word (everything after the last char belonging to this frame) is stored
        in ``_overflow_word``, so the caller can attribute it to the next
        AggregatedTextFrame.

        Args:
            word: A single word token returned by the TTS service. TTS services that
                emit spaces and punctuation as separate tokens (e.g. Inworld) must
                pre-merge those tokens into the preceding word before calling this
                method (see ``TTSService._merge_punct_tokens``).

        Returns:
            True when all expected content has been covered.
        """
        normalized = self._normalize(word)

        prev_len = len(self._received)
        expected_len = len(self._tts_normalized)

        self._overflow_word = None
        self._llm_consumed = None
        self._frame_word = None

        if prev_len > expected_len:
            logger.warning(f"{self}, trying to add a word in an already complete frame")
            return True

        # If the word doesn't match the next expected chars, the TTS provider
        # likely dropped a word-timestamp event. Force-complete this slot: emit the
        # remaining TTS text as _frame_word so a TTSTextFrame is still produced
        # for the unspoken portion, consume all remaining llm_text, and route the
        # entire incoming word as overflow for the next slot.
        if not self.word_belongs_here(word):
            self._frame_word = self._tts_text[self._tts_pos :]
            if self._llm_text is not None:
                self._llm_consumed = self._llm_text[self._llm_pos :]
                self._llm_pos = len(self._llm_text)
                # This should not happen: force-complete sweeps all remaining
                # llm_text, so the span must contain the frame word. If it
                # doesn't, tts_text and llm_text are out of sync in an
                # unexpected way — discard rather than returning a corrupt span.
                # Also removing punctuation from the frame word to match the
                # expected text, since some TTS services may add punctuation to
                # the raw text.
                word_without_punctuation = self._remove_trailing_punctuation(self._frame_word)
                if word_without_punctuation and word_without_punctuation not in self._llm_consumed:
                    logger.warning(
                        f"WordCompletionTracker: force-complete llm_consumed {repr(self._llm_consumed)!s} "
                        f"does not contain frame_word {repr(self._frame_word)!s}, discarding"
                    )
                    self._llm_consumed = None
            self._received = self._tts_normalized  # force-complete
            self._overflow_word = word
            return True

        self._received += normalized

        # How many normalized chars from this word belong to the current frame.
        chars_for_frame = min(len(normalized), expected_len - prev_len)

        if prev_len + len(normalized) > expected_len:
            # This word straddles the frame boundary. Split into:
            #   - _frame_word: the prefix of `word` up to the split point, used
            #     for the TTSTextFrame of the current slot.
            #   - raw overflow word: the raw suffix after the split point, used
            #     to build a TTSTextFrame attributed to the next AggregatedTextFrame.
            split_pos = self._advance_by_alnums(word, 0, chars_for_frame)
            self._frame_word = word[:split_pos]
            self._overflow_word = word[split_pos:]
        else:
            # Word fits entirely in this frame.
            self._frame_word = word

        # Advance the TTS cursor by the same alnum count so the force-complete
        # path knows where in _tts_text to start from.
        self._tts_pos = self._advance_by_alnums(self._tts_text, self._tts_pos, chars_for_frame)

        if self._llm_text is not None:
            if self.is_complete:
                # Consume ALL remaining LLM text: closing tags (e.g. </card>)
                # and any trailing punctuation that the TTS will not send separately.
                self._llm_consumed = self._llm_text[self._llm_pos :]
                self._llm_pos = len(self._llm_text)
            else:
                if chars_for_frame == 0:
                    # Consume exactly the raw word in llm_text, skipping any
                    # leading spaces that belong to the previous token's span.
                    start = self._llm_pos
                    while start < len(self._llm_text) and self._llm_text[start].isspace():
                        start += 1
                    end = start + len(word)
                    self._llm_consumed = self._llm_text[start:end]
                    self._llm_pos = end
                else:
                    # Advance through llm_text by exactly chars_for_frame alphanumeric
                    # chars. Non-alnum chars (spaces, opening tags) are included in the
                    # slice, preserving the original formatting for the context.
                    new_pos = self._advance_by_alnums(
                        self._llm_text, self._llm_pos, chars_for_frame
                    )
                    self._llm_consumed = self._llm_text[self._llm_pos : new_pos]
                    self._llm_pos = new_pos
            # This should not happen: the LLM cursor is driven by the same
            # alnum count as the word stream, so the consumed span must contain
            # the frame word. If it doesn't, the cursors drifted out of sync
            # in an unexpected way — discard rather than returning a corrupt span.
            # Also removing punctuation from the frame word to match the
            # expected text, since some TTS services may add punctuation to
            # the raw text.
            word_without_punctuation = self._remove_trailing_punctuation(self._frame_word)
            if word_without_punctuation and self._fold_typography(
                word_without_punctuation
            ) not in self._fold_typography(self._llm_consumed):
                logger.warning(
                    f"WordCompletionTracker: llm_consumed {repr(self._llm_consumed)!s} "
                    f"does not contain frame_word {repr(self._frame_word)!s}, discarding"
                )
                self._llm_consumed = None

        return self.is_complete

    def word_belongs_here(self, word: str) -> bool:
        """Return True if this word plausibly belongs to the remaining TTS text.

        Dispatches to one of two checks depending on whether the word contains
        any alphanumeric characters after normalization:

        - Alnum words: prefix-match against the remaining expected chars.
        - Symbol/punctuation words (empty after normalization): literal substring
          search in the remaining raw TTS text, with a fallback for TTS providers
          that substitute Unicode symbols with ASCII punctuation.

        Used to detect when the TTS provider silently dropped a word-timestamp
        event: if the incoming word does not match this slot's remaining content,
        the caller should force-complete this slot and route the word to the next.
        """
        normalized = self._normalize(word)
        if normalized:
            return self._alnum_word_belongs_here(normalized)
        else:
            return self._symbol_word_belongs_here(word)

    def _alnum_word_belongs_here(self, normalized: str) -> bool:
        """Return True if an alnum-containing word matches this frame's remaining expected chars.

        Accepts both full words and partial tokens — the word belongs here as long
        as its normalized characters are a prefix of what is still expected. This
        also handles the overflow case where the word is longer than the remaining
        content (the excess is detected and split in ``add_word_and_check_complete``).
        """
        remaining = self._tts_normalized[len(self._received) :]
        if not remaining:
            return False
        check_len = min(len(normalized), len(remaining))
        return remaining.startswith(normalized[:check_len])

    def _symbol_word_belongs_here(self, word: str) -> bool:
        """Return True if a non-alnum word (emoji, punctuation, symbol) belongs to this frame.

        Two checks are applied in order:

        1. **Literal substring**: search for the raw word in the remaining TTS text.
           ``_advance_by_alnums`` may have already moved ``_tts_pos`` past some trailing
           punctuation, so the search window is backed up to include those characters.

        2. **Symbol substitution fallback**: some TTS providers substitute Unicode symbols
           with ASCII punctuation in word-timestamp events (e.g. ElevenLabs reports ``→``
           as ``-``), so check 1 always fails even though the word belongs here. If alnum
           content still remains unconsumed and the next non-space character in the TTS
           text is itself a non-alnum symbol, accept the word as a substitution.
        """
        search_start = self._tts_pos
        while search_start > 0:
            ch = self._tts_text[search_start - 1]
            if ch.isalnum() or ch.isspace() or ch == ">":
                break
            search_start -= 1
        if word in self._tts_text[search_start:]:
            return True

        if len(self._received) >= len(self._tts_normalized):
            return False

        pos = self._tts_pos
        while pos < len(self._tts_text) and self._tts_text[pos].isspace():
            pos += 1
        return pos < len(self._tts_text) and not self._tts_text[pos].isalnum()

    def get_word_for_frame(self) -> str | None:
        """Return the portion of the last word that belongs to this frame.

        - Normal word (no overflow): the full word.
        - Straddling word: the prefix up to the frame boundary (e.g. ``"1111"``
          from ``"1111 And"``).
        - Force-completed (word didn't belong): the remaining unspoken text from
          ``tts_text`` so a TTSTextFrame can still be emitted for the dropped
          portion. The incoming word is routed as overflow to the next slot.
        """
        return self._frame_word.strip() if self._frame_word else self._frame_word

    def get_overflow_word(self) -> str | None:
        """Return the raw suffix of the last word that overflows into the next frame.

        Preserves the original casing and any non-alphanumeric characters so the
        overflow TTSTextFrame has natural word text. Returns None when there is no
        overflow (the word fit entirely within this frame).
        """
        return self._overflow_word.strip() if self._overflow_word else self._overflow_word

    def get_llm_consumed(self) -> str | None:
        """Return the LLM text span consumed for the last added word.

        Returns None if no llm_text was provided at construction time.
        """
        return self._llm_consumed.strip() if self._llm_consumed else self._llm_consumed

    def get_accumulated_tts_text(self) -> str:
        """Return all consumed text from tts_text up to the current cursor position.

        Unlike ``get_word_for_frame()`` (which reflects only the last word), this returns
        everything that has been consumed since construction or the last ``reset()``.
        """
        return self._tts_text[: self._tts_pos]

    def get_accumulated_llm_text(self) -> str | None:
        """Return all consumed text from llm_text up to the current cursor position.

        Unlike ``get_llm_consumed()`` (which reflects only the last word), this returns
        everything that has been consumed since construction or the last ``reset()``.
        Returns None if no llm_text was provided at construction time.
        """
        if self._llm_text is None:
            return None
        return self._llm_text[: self._llm_pos]

    def get_remaining_tts_text(self) -> str:
        """Return the unspoken portion of tts_text, stripped of leading/trailing whitespace.

        This is the text that the TTS provider has not yet confirmed via word-timestamp
        events. Useful for force-completing a slot when the audio context ends before all
        word-timestamp events have arrived.
        """
        return self._tts_text[self._tts_pos :].strip()

    def get_remaining_llm_text(self) -> str | None:
        """Return the unspoken portion of llm_text, stripped of leading/trailing whitespace.

        Returns None if no llm_text was provided at construction time. Like
        ``get_remaining_tts_text()``, intended for force-completing a slot so that the
        conversation context receives the full original text.
        """
        if self._llm_text is None:
            return None
        remaining = self._llm_text[self._llm_pos :].strip()
        return remaining if remaining else None

    @property
    def is_complete(self) -> bool:
        """True when accumulated normalized chars >= expected normalized chars."""
        return len(self._received) >= len(self._tts_normalized)

    def reset(self):
        """Reset received word accumulation without changing the expected text."""
        self._received = ""
        self._tts_pos = 0
        self._llm_pos = 0
        self._overflow_word = None
        self._llm_consumed = None
        self._frame_word = None
