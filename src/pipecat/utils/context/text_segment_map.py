#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Segment-level mapping between original and TTS-transformed text."""

import difflib
import re
from dataclasses import dataclass

from pipecat.utils.text.transforms._alnum_utils import advance_by_alnums, normalize


def _strip_tag_markup(word: str, in_open_tag: bool) -> tuple[str, bool]:
    """Strip SSML/XML tag markup from *word*, tracking tags that span words.

    A tag whose opening tag contains internal whitespace (e.g. multiple
    attributes, as in ElevenLabs' ``<phoneme alphabet="..." ph="...">``) can be
    reported as several separate word-timestamp tokens by TTS providers that
    tokenize on whitespace without tag awareness. This walks *word*
    character-by-character, carrying open/close state across calls so a tag
    opened in one word and closed in a later one (or an attribute-only word
    entirely inside an open tag) is still recognised as pure markup.

    Args:
        word: Raw word token to strip.
        in_open_tag: Whether a previous word left an unclosed tag open.

    Returns:
        (content, still_in_open_tag): *word* with all tag markup removed, and
        whether a tag remains open after processing it.
    """
    result = []
    i, n = 0, len(word)
    while i < n:
        if in_open_tag:
            close = word.find(">", i)
            if close == -1:
                i = n
            else:
                in_open_tag = False
                i = close + 1
        else:
            open_ = word.find("<", i)
            if open_ == -1:
                result.append(word[i:])
                i = n
            else:
                result.append(word[i:open_])
                close = word.find(">", open_)
                if close == -1:
                    in_open_tag = True
                    i = n
                else:
                    i = close + 1
    return "".join(result), in_open_tag


@dataclass(frozen=True)
class TextSegment:
    """Immutable aligned chunk between original and TTS text.

    Parameters:
        original: Chunk of the user-facing / LLM text.
        tts: Corresponding chunk in the TTS-transformed text.
        original_start: Byte offset in original_text where this chunk begins.
        original_end: Byte offset in original_text where this chunk ends.
    """

    original: str
    tts: str
    original_start: int
    original_end: int

    @property
    def is_transformed(self) -> bool:
        """True when this segment cannot be tracked by proportional char advancement.

        This holds either when the alphanumeric content differs between original
        and TTS sides, or when a replacement changed the segment's word count
        (e.g. splitting ``"BODYPUMP"`` into ``"body pump"``, or letter-spacing an
        acronym like ``"API"`` into ``"A P I"``). Word-splitting replacements
        can normalize to the same alphanumeric content on both sides, but the
        proportional advance still breaks: it would consume alnum chars from a
        single contiguous original token using word boundaries that only exist
        on the TTS side, landing mid-word instead of at a real boundary.
        """
        if normalize(self.original) != normalize(self.tts):
            return True
        return len(self.original.split()) != len(self.tts.split())

    @property
    def tts_alnum_count(self) -> int:
        """Number of alphanumeric characters in the TTS side of this segment."""
        return len(normalize(self.tts))

    @property
    def original_alnum_count(self) -> int:
        """Number of alphanumeric characters in the original side of this segment."""
        return len(normalize(self.original))


class TextSegmentMap:
    """Maps cursor positions between transformed TTS text and original text.

    Built once from two texts that may differ in alphanumeric content due to text
    transforms (e.g. currency expansion). Tracks how many TTS alnum chars have been
    consumed word-by-word and exposes the corresponding position in the original text.

    For unchanged segments, both cursors advance proportionally. For transformed
    segments, both cursors are held until the entire TTS segment has been consumed,
    then jump to the end of the corresponding original segment in one step.

    Callers advance the map word-by-word via :meth:`advance_word`, which also
    strips SSML tag markup from a raw word-timestamp token, tracking tags whose
    opening tag spans multiple words (e.g. an SSML tag with several attributes,
    split on whitespace by some TTS providers' word-timestamp streams).

    Example::

        # "$42.50" was expanded to "forty two dollars and fifty cents"
        smap = TextSegmentMap(
            "Your balance is forty two dollars and fifty cents",
            "Your balance is $42.50",
        )
        for word in ["Your", "balance", "is"]:
            smap.advance_word(word)   # unchanged segment
        for word in ["forty", "two", "dollars", "and", "fifty"]:
            smap.advance_word(word)   # transformed segment, cursors held
        smap.advance_word("cents")    # segment completes, cursors jump
        assert smap.last_completed_segment.original == "$42.50"
        assert not smap.in_transformed_segment
    """

    def __init__(
        self,
        tts_text: str,
        original_text: str,
        llm_text: str | None = None,
    ):
        """Initialize the segment map.

        Args:
            tts_text: Post-transform text sent to TTS.
            original_text: User-facing pre-transform text (no surrounding tags).
            llm_text: LLM-produced text, which may have surrounding tags like
                ``<card>...</card>``. Defaults to ``original_text`` when not provided.
                The LLM cursor advances through this text at the same segment
                boundaries as the user-facing cursor.
        """
        self._original_text = original_text
        self._llm_text = llm_text if llm_text is not None else original_text
        self._segments: list[TextSegment] = self._build(tts_text, original_text)
        self._reset_state()

    @staticmethod
    def _build(tts_text: str, original_text: str) -> list[TextSegment]:
        """Build aligned TextSegments from a word-level SequenceMatcher diff.

        Each diff opcode (equal, replace, insert, delete) becomes a segment.
        Segments whose normalized alphanumeric content differs are later treated
        as transformed/atomic units during cursor advancement.
        """

        def tokenize(text: str) -> list[str]:
            return re.split(r"(\s+)", text)

        orig_tokens = tokenize(original_text)
        tts_tokens = tokenize(tts_text)

        # SequenceMatcher produces a word-level alignment between the original
        # and TTS texts. Each opcode becomes a TextSegment whose boundaries are
        # tracked in the original text.
        #
        # Example:
        #
        #     original_text = "Your balance is $42.50"
        #     tts_text      = "Your balance is forty two dollars and fifty cents"
        #
        # Tokenization preserves whitespace, so SequenceMatcher sees:
        #
        #     equal:
        #         "Your balance is "
        #
        #     replace:
        #         "$42.50"
        #         ->
        #         "forty two dollars and fifty cents"
        #
        # This produces two segments:
        #
        #     TextSegment(
        #         original="Your balance is ",
        #         tts="Your balance is ",
        #         original_start=0,
        #         original_end=16,
        #     )
        #
        #     TextSegment(
        #         original="$42.50",
        #         tts="forty two dollars and fifty cents",
        #         original_start=16,
        #         original_end=22,
        #     )
        #
        # During playback, unchanged segments advance cursors
        # proportionally. Transformed segments are treated as atomic:
        # the cursors are held while the expanded TTS text is being
        # consumed and jump to original_end only when the entire
        # transformed segment completes.
        matcher = difflib.SequenceMatcher(None, orig_tokens, tts_tokens, autojunk=False)

        segments: list[TextSegment] = []
        orig_pos = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            orig_chunk = "".join(orig_tokens[i1:i2])
            tts_chunk = "".join(tts_tokens[j1:j2])
            orig_end = orig_pos + len(orig_chunk)
            segments.append(
                TextSegment(
                    original=orig_chunk,
                    tts=tts_chunk,
                    original_start=orig_pos,
                    original_end=orig_end,
                )
            )
            orig_pos = orig_end

        return segments

    def _reset_state(self) -> None:
        self._seg_idx: int = 0
        self._seg_consumed: int = 0
        self._user_facing_pos: int = 0
        self._llm_pos: int = 0
        self._last_completed: TextSegment | None = None
        self._touched_current_segment: bool = False
        self._in_open_tag: bool = False

    def strip_word(self, word: str) -> str:
        """Return *word* with any SSML tag markup removed, without consuming it.

        Non-mutating peek variant of :meth:`advance_word`'s stripping step, so
        callers can decide whether a word belongs here before consuming it.

        Args:
            word: Raw word token to strip.

        Returns:
            *word* with tag markup removed.
        """
        content, _ = _strip_tag_markup(word, self._in_open_tag)
        return content

    def advance_word(self, word: str) -> str:
        """Strip tag markup from *word*, commit the tag state, and advance.

        Combines :meth:`strip_word` with the internal char-count advance: strips
        *word* (tracking any tag left open for the next call), then advances
        cursors by the resulting content's alphanumeric character count -- zero
        for a word that is entirely tag markup.

        Args:
            word: Raw TTS word-timestamp token.

        Returns:
            *word* with tag markup removed. Callers should use this in place of
            the raw word for their own alnum accounting.
        """
        content, self._in_open_tag = _strip_tag_markup(word, self._in_open_tag)
        self._advance(len(normalize(content)))
        return content

    def _advance(self, n_alnum: int) -> None:
        """Consume *n_alnum* TTS alphanumeric chars, advancing internal cursors.

        For unchanged segments the cursors move proportionally through both original
        and LLM text. For transformed segments the cursors are held until the whole
        segment is consumed, then jump to the end of the original segment.

        Internal primitive behind :meth:`advance_word`, which is the entry point
        callers should use -- it strips SSML tag markup first, which a bare
        alnum count can't account for on its own.

        Args:
            n_alnum: Number of TTS alphanumeric characters to consume.
        """
        self._last_completed = None
        seg_idx_before = self._seg_idx

        # A segment can require zero TTS alnum chars (e.g. an inline IPA tag
        # that normalizes to no alnum content). Such a segment never has
        # anything for the loop below to consume, so it can only complete
        # here, once -- typically on the very call whose own word normalizes
        # to zero chars too (n_alnum == 0).
        if self._seg_idx < len(self._segments):
            seg = self._segments[self._seg_idx]
            if seg.tts_alnum_count - self._seg_consumed == 0:
                self._complete_or_advance_segment(consume=0)

        remaining = n_alnum
        while remaining > 0 and self._seg_idx < len(self._segments):
            seg = self._segments[self._seg_idx]
            available = seg.tts_alnum_count - self._seg_consumed
            consume = min(remaining, available)
            remaining -= consume
            self._complete_or_advance_segment(consume)

        # True once this call has processed a word without moving off of
        # seg_idx_before via completion -- i.e. the cursor's current segment
        # is one this call actually touched, as opposed to one it merely
        # landed on by completing the previous segment.
        self._touched_current_segment = self._seg_idx == seg_idx_before

    def _complete_or_advance_segment(self, consume: int) -> None:
        """Apply *consume* alnum chars to the segment at the current index.

        Completes and advances to the next segment if its budget is now fully
        spent; otherwise advances an in-progress unchanged segment's cursors
        proportionally (a transformed segment in progress just holds).
        """
        seg = self._segments[self._seg_idx]
        self._seg_consumed += consume

        if self._seg_consumed == seg.tts_alnum_count:
            if seg.is_transformed:
                # Transformed: snap user_facing_pos to the end of the original
                # pattern and jump llm_pos by the full original_alnum_count from
                # where it was held during the segment.
                self._user_facing_pos = seg.original_end
                self._llm_pos = advance_by_alnums(
                    self._llm_text, self._llm_pos, seg.original_alnum_count
                )
            else:
                # Unchanged: advance both cursors proportionally for the final
                # call (same as the in-progress path). Using advance_by_alnums
                # instead of seg.original_end avoids overshooting when a segment
                # ends with trailing whitespace (e.g. " 1111 1111 ").
                self._user_facing_pos = advance_by_alnums(
                    self._original_text, self._user_facing_pos, consume
                )
                self._llm_pos = advance_by_alnums(self._llm_text, self._llm_pos, consume)
            self._last_completed = seg
            self._seg_idx += 1
            self._seg_consumed = 0
        elif not seg.is_transformed:
            # Unchanged segment in progress: advance both cursors proportionally.
            self._user_facing_pos = advance_by_alnums(
                self._original_text, self._user_facing_pos, consume
            )
            self._llm_pos = advance_by_alnums(self._llm_text, self._llm_pos, consume)
        # else: transformed segment in progress — hold both cursors.

    @property
    def user_facing_pos(self) -> int:
        """Current byte offset in the original (user-facing) text."""
        return self._user_facing_pos

    @property
    def llm_pos(self) -> int:
        """Current byte offset in the LLM text."""
        return self._llm_pos

    @property
    def in_transformed_segment(self) -> bool:
        """True when the cursor is on a transformed segment that isn't complete yet.

        True once the segment's alnum budget has partly been consumed (the
        original condition), or once a call has touched this segment without
        consuming anything (e.g. a leading zero-alnum fragment such as a
        still-open tag's attribute text, which normalizes to ``""``) -- as long
        as that call didn't simply land here by completing the *previous*
        segment, in which case the word that triggered it belongs to that
        previous segment, not this one.
        """
        if self._seg_idx >= len(self._segments):
            return False
        seg = self._segments[self._seg_idx]
        return seg.is_transformed and (self._seg_consumed > 0 or self._touched_current_segment)

    @property
    def last_completed_segment(self) -> TextSegment | None:
        """The segment completed by the last :meth:`advance_word` call, or ``None``."""
        return self._last_completed

    def reset(self) -> None:
        """Reset all cursor and consumption state to initial values."""
        self._reset_state()
