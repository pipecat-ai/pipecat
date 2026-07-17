#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Word completion tracker for TTS context ordering."""

import re
import unicodedata

from loguru import logger

from pipecat.utils.context.text_segment_map import TextSegmentMap, strip_complete_markup


class WordCompletionTracker:
    """Tracks whether all words from a source AggregatedTextFrame have been spoken.

    Delegates completion tracking and cursor advancement entirely to a
    :class:`~pipecat.utils.context.text_segment_map.TextSegmentMap` built from
    ``tts_text`` (which may include TTS-specific SSML tags, e.g. ``<spell>...</spell>``
    returned by some TTS providers in word-timestamp events). The map matches each
    incoming word against the remaining TTS text and reports when the frame is
    fully spoken, robust to punctuation, spacing, and markup -- this tracker's own
    bookkeeping is limited to overriding cursors when a slot is force-completed
    (see below).

    When ``llm_text`` is provided (e.g. the original pattern-matched text including
    delimiters like ``<card>4111 1111 1111 1111</card>``), the tracker additionally
    maps each spoken word back to its corresponding span in that LLM text. This
    lets callers attach the original text to ``TTSTextFrame`` entries so the
    conversation context receives properly-tagged content rather than the cleaned
    words received from the TTS provider.

    For unchanged segments (no text transforms applied) both cursors advance
    proportionally word-by-word; for transformed segments (e.g. ``"$42.50"`` →
    ``"forty two dollars and fifty cents"``) both cursors are held until the entire
    TTS segment is consumed, then jump to the end of the original span in one step.

    Background: TTS providers apply their own SSML tags to the text before
    synthesis and return word-timestamp events containing the raw spoken words
    (e.g. ``"4111"``, ``"1111"``). Without LLM-text tracking, the conversation
    context would only see those cleaned words and lose the original structure
    (e.g. ``<card>4111 1111 1111 1111</card>``). By mapping consumed spans back
    to positions in ``llm_text``, each TTSTextFrame can carry the exact span of
    original text it represents.

    Overflow handling: TTS providers sometimes return a single word token that
    spans the boundary between two AggregatedTextFrames (e.g. ``"1111</spell>And"``
    when one frame ends with ``1111</card>`` and the next begins with ``And``). The
    tracker detects this and exposes the raw overflow suffix via ``get_overflow_word()``,
    so callers can feed the remainder into the next frame's tracker and emit a
    correctly-attributed TTSTextFrame for each part.

    Example::

        tracker = WordCompletionTracker("Hello, world!")
        tracker.add_word_and_check_complete("Hello")   # False
        tracker.add_word_and_check_complete("world")   # True  — all TTS text consumed
    """

    def __init__(
        self,
        tts_text: str,
        llm_text: str | None = None,
        user_facing_text: str | None = None,
    ):
        """Initialize the tracker with the text of the frame being spoken.

        Args:
            tts_text: Full text of the AggregatedTextFrame sent to TTS (may include
                TTS-specific SSML tags). Used as the cursor reference for the TTS
                word stream.
            llm_text: Original LLM-produced text including pattern delimiters (e.g.
                ``<card>4111 1111 1111 1111</card>``). When provided, each
                ``add_word_and_check_complete`` call also returns the corresponding
                LLM span via ``get_llm_consumed()``.
            user_facing_text: The original text of the AggregatedTextFrame as shown
                to the user (e.g. via RTVI). Unlike ``tts_text``, this text has no
                TTS-specific tags or transformations. The tracker maintains a cursor
                into it so callers can retrieve the spoken and unspoken portions in
                terms of user-visible text via ``get_accumulated_user_facing_text()``
                and ``get_remaining_user_facing_text()``. Defaults to ``tts_text``
                with markup stripped when not provided -- user-facing text should
                never carry synthesis tags.
        """
        # _tts_text is the raw text sent to TTS (may carry SSML tags). The segment
        # map's raw_pos indexes into it; the get_*_tts_text accessors slice it.
        self._tts_text = tts_text

        # _user_facing_text is the original text returned to the user (e.g. via RTVI).
        # Falls back to tts_text (markup stripped) when not provided so this cursor
        # is always valid, and so the segment map still splits out a non-tagged
        # prefix/suffix around any markup instead of treating the whole identical
        # string as one big segment.
        # _user_facing_pos is a cursor into it, kept in sync with the segment map
        # except when a slot is force-completed (which the segment map never
        # observes, since it manually jumps this cursor to the end).
        self._user_facing_text: str = (
            user_facing_text if user_facing_text is not None else strip_complete_markup(tts_text)
        )
        self._user_facing_pos = 0

        # _llm_text is the original LLM-produced text (with pattern delimiters like
        # <card>...</card>). _llm_pos is a cursor into it, kept in sync with the
        # segment map the same way as _user_facing_pos.
        self._llm_text = llm_text
        self._llm_pos = 0

        # Per-call outputs: recomputed on every add_word_and_check_complete and
        # read back through the get_* accessors. _frame_word is the part of the
        # last word belonging to this frame; _overflow_word is the part that
        # spilled into the next frame; _llm_consumed is the llm_text span it maps to.
        self._overflow_word: str | None = None
        self._llm_consumed: str | None = None
        self._frame_word: str | None = None

        # Set when a slot is force-completed (a word didn't match the remaining
        # TTS text, e.g. the provider dropped a word-timestamp event). The segment
        # map itself is never advanced in that case, so its own is_complete stays
        # stale -- this flag is the authoritative completion signal from then on.
        self._force_completed = False

        self._segment_map = TextSegmentMap(tts_text, self._user_facing_text, llm_text)

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
    def _fold_for_comparison(text: str) -> str:
        """Fold text for lenient span-containment comparisons.

        Applies typographic folding, casefolds, and collapses connector
        characters (spaces and hyphens). This makes the comparison tolerant of
        case-only replacements (``"SQL"`` vs ``"sql"``) and replacements that
        only change how words are joined (``"BODYPUMP"`` vs ``"body-pump"``),
        while still preserving other content (digits, emoji, punctuation) so
        the safeguard can detect a genuinely missing/mismatched word.
        """
        folded = WordCompletionTracker._fold_typography(text).casefold()
        return re.sub(r"[-\s]+", "", folded)

    @staticmethod
    def _remove_trailing_punctuation(text: str) -> str:
        """Remove punctuation only at the very end of the given text."""
        i = len(text)
        while i > 0 and unicodedata.category(text[i - 1]).startswith("P"):
            i -= 1
        return text[:i]

    def add_word_and_check_complete(self, word: str) -> bool:
        """Record a spoken word from a word-timestamp event.

        Before advancing, checks whether the word belongs to this frame via
        ``word_belongs_here``. If it does not (e.g. the TTS provider silently
        dropped a word-timestamp), the slot is force-completed: the remaining
        unspoken text from ``tts_text`` is stored in ``_frame_word`` so a
        TTSTextFrame can still be emitted for the dropped portion, all remaining
        ``llm_text`` is consumed, and the entire incoming word is set as overflow
        so the caller's overflow path routes it to the next slot unchanged.

        Otherwise the word is handed to the segment map, which matches it against
        the remaining TTS text and advances its own cursors. If ``llm_text`` was
        provided at construction time, also stores the corresponding LLM span in
        ``_llm_consumed``. When this word completes the frame, the entire remaining
        LLM text (including any closing tags) is consumed so nothing is lost.

        If the word overshoots the expected length (overflow -- it spans the
        boundary into the next AggregatedTextFrame), the raw suffix of the word is
        stored in ``_overflow_word``, so the caller can attribute it to the next frame.

        Args:
            word: A single word token returned by the TTS service. TTS services that
                emit spaces and punctuation as separate tokens (e.g. Inworld) must
                pre-merge those tokens into the preceding word before calling this
                method (see ``TTSService._merge_punct_tokens``). May also be a
                fragment of a still-open SSML tag; the segment map matches such
                fragments against the remaining TTS text without needing to parse
                them as markup.

        Returns:
            True when all expected content has been covered.
        """
        self._overflow_word = None
        self._llm_consumed = None
        self._frame_word = None

        # Reject only once every raw char of tts_text has actually been consumed.
        # `is_complete` (alnum-based) can turn True earlier -- e.g. a frame ending
        # in a symbol/emoji that contributes no alphanumeric content is "complete"
        # before that trailing word arrives -- but such a word must still be
        # accepted normally rather than rejected here.
        if self._force_completed or self._segment_map.raw_pos >= len(self._tts_text):
            logger.warning(f"{self}, trying to add a word in an already complete frame")
            return True

        # If the word doesn't match the next expected text, the TTS provider
        # likely dropped a word-timestamp event. Force-complete this slot: emit the
        # remaining TTS text as _frame_word so a TTSTextFrame is still produced
        # for the unspoken portion, consume all remaining llm_text, and route the
        # entire incoming word as overflow for the next slot.
        if not self.word_belongs_here(word):
            self._frame_word = self._tts_text[self._segment_map.raw_pos :]
            self._user_facing_pos = len(self._user_facing_text)
            if self._llm_text is not None:
                # Sweep all remaining llm_text so nothing is lost, then guard
                # against a tts_text/llm_text desync (see the helper).
                self._llm_consumed = self._llm_text[self._llm_pos :]
                self._llm_pos = len(self._llm_text)
                self._discard_llm_span_if_frame_word_missing()
            self._force_completed = True
            self._overflow_word = word
            return True

        # Word belongs to this frame: let the segment map match it against the
        # remaining TTS text and advance its own cursors.
        prev_llm_pos = self._llm_pos
        self._segment_map.advance_word(word)

        overflow = self._segment_map.last_overflow
        self._frame_word = word[: len(word) - len(overflow)] if overflow else word
        self._overflow_word = overflow

        self._user_facing_pos = self._segment_map.user_facing_pos
        self._llm_pos = self._segment_map.llm_pos

        if self._llm_text is not None:
            self._attribute_llm_consumed(word, prev_llm_pos)

        return self.is_complete

    def _attribute_llm_consumed(self, word: str, prev_llm_pos: int) -> None:
        """Set ``_llm_consumed`` to the llm_text span the just-advanced word maps to.

        Only called when ``llm_text`` was provided. Four cases:

        - **Frame completed**: sweep everything from *prev_llm_pos* to the end so
          the completing word's span plus any trailing closing tags (e.g.
          ``</card>``) are included.
        - **Mid transformed segment**: attribution is suppressed (``None``) --
          only the word that completes the segment carries its original text.
        - **Non-alnum word** (emoji/punctuation) that moved no cursor and
          completed no segment: consume its raw characters from ``llm_text``
          directly, skipping leading spaces owned by the previous token.
        - **Otherwise**: the span from *prev_llm_pos* to the new llm cursor
          (covers a normal word, or a zero-budget segment that completed via this
          word since ``llm_pos`` was already synced from its jump).

        Except mid-transformed-segment, the span is validated against the frame
        word and discarded on a desync (see
        :meth:`_discard_llm_span_if_frame_word_missing`). The validation is
        skipped when the completing word finished a transformed segment, since
        the spoken word (e.g. ``"dollars"``) won't appear verbatim in the
        original (e.g. ``"$5"``).
        """
        assert self._llm_text is not None
        completed = self._segment_map.last_completed_segment

        if self.is_complete:
            self._llm_consumed = self._llm_text[prev_llm_pos:]
            self._llm_pos = len(self._llm_text)
            if completed is None or not completed.is_transformed:
                self._discard_llm_span_if_frame_word_missing()
        elif self._segment_map.in_transformed_segment:
            self._llm_consumed = None
        elif self._llm_pos == prev_llm_pos and completed is None:
            start = self._llm_pos
            while start < len(self._llm_text) and self._llm_text[start].isspace():
                start += 1
            end = start + len(word)
            self._llm_consumed = self._llm_text[start:end]
            self._llm_pos = end
        else:
            self._llm_consumed = self._llm_text[prev_llm_pos : self._llm_pos]
            if completed is None or not completed.is_transformed:
                self._discard_llm_span_if_frame_word_missing()

    def _discard_llm_span_if_frame_word_missing(self) -> None:
        """Drop ``_llm_consumed`` if it doesn't contain the current frame word.

        A safeguard against ``tts_text`` and ``llm_text`` drifting out of sync:
        the span attributed to a word should contain that word. Compared case-
        and connector-insensitively (casefolded, hyphens/spaces collapsed) so a
        case-only (``"SQL"`` vs ``"sql"``) or hyphen-vs-space replacement isn't
        mistaken for a desync. An all-punctuation frame word (nothing to match)
        is left alone.
        """
        frame_word = self._remove_trailing_punctuation(self._frame_word or "")
        if not frame_word:
            return
        if self._fold_for_comparison(frame_word) in self._fold_for_comparison(
            self._llm_consumed or ""
        ):
            return
        logger.warning(
            f"WordCompletionTracker: llm_consumed {repr(self._llm_consumed)!s} "
            f"does not contain frame_word {repr(self._frame_word)!s}, discarding"
        )
        self._llm_consumed = None

    def word_belongs_here(self, word: str) -> bool:
        """Return True if this word plausibly belongs to the remaining TTS text.

        Delegates entirely to the segment map, which owns the remaining-text
        matching needed to decide.

        Used to detect when the TTS provider silently dropped a word-timestamp
        event: if the incoming word does not match this slot's remaining content,
        the caller should force-complete this slot and route the word to the next.
        """
        return self._segment_map.word_belongs_current_segment(word)

    def suppress_in_context(self) -> bool:
        """True when the last word is mid-flight inside a transformed segment.

        When True, the sequencer sets ``append_to_context=False`` on the emitted
        ``TTSTextFrame`` so intermediate TTS words (e.g. "forty", "two") are not
        written to the conversation context. Only the completing word of the segment
        carries ``raw_text`` with the original text (e.g. ``"$42.50"``).
        """
        return self._segment_map.in_transformed_segment

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

    def get_accumulated_user_facing_text(self) -> str:
        """Return all consumed text from user_facing_text up to the current cursor position."""
        return self._user_facing_text[: self._user_facing_pos]

    def get_remaining_user_facing_text(self, strip: bool = True) -> str:
        """Return the unspoken portion of user_facing_text.

        Args:
            strip: When True (default), leading/trailing whitespace is removed.
                Set to False to preserve leading whitespace so that
                ``get_accumulated_user_facing_text() + get_remaining_user_facing_text(strip=False)``
                reconstructs the original text exactly.
        """
        remaining = self._user_facing_text[self._user_facing_pos :]
        return remaining.strip() if strip else remaining

    def get_accumulated_tts_text(self) -> str:
        """Return all consumed text from tts_text up to the current cursor position.

        Unlike ``get_word_for_frame()`` (which reflects only the last word), this returns
        everything that has been consumed since construction or the last ``reset()``.
        """
        return self._tts_text[: self._segment_map.raw_pos]

    def get_accumulated_llm_text(self) -> str | None:
        """Return all consumed text from llm_text up to the current cursor position.

        Unlike ``get_llm_consumed()`` (which reflects only the last word), this returns
        everything that has been consumed since construction or the last ``reset()``.
        Returns None if no llm_text was provided at construction time.
        """
        if self._llm_text is None:
            return None
        return self._llm_text[: self._llm_pos]

    def get_remaining_tts_text(self, strip: bool = True) -> str:
        """Return the unspoken portion of tts_text.

        Args:
            strip: When True (default), leading/trailing whitespace is removed.
                Set to False to preserve leading whitespace so that
                ``get_accumulated_tts_text() + get_remaining_tts_text(strip=False)``
                reconstructs the original text exactly.
        """
        remaining = self._tts_text[self._segment_map.raw_pos :]
        return remaining.strip() if strip else remaining

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
        """True when this frame's TTS text has been fully accounted for."""
        return self._force_completed or self._segment_map.is_complete

    def reset(self):
        """Reset all cursors and per-call outputs without changing the expected texts."""
        self._user_facing_pos = 0
        self._llm_pos = 0
        self._overflow_word = None
        self._llm_consumed = None
        self._frame_word = None
        self._force_completed = False
        self._segment_map.reset()
