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
        """True when the alphanumeric content differs between original and TTS sides."""
        return normalize(self.original) != normalize(self.tts)

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

    Example::

        # "$42.50" was expanded to "forty two dollars and fifty cents"
        smap = TextSegmentMap(
            "Your balance is forty two dollars and fifty cents",
            "Your balance is $42.50",
        )
        smap.advance(4)   # "Your" — unchanged segment
        smap.advance(7)   # "balance" — unchanged segment
        smap.advance(2)   # "is" — unchanged segment
        smap.advance(5)   # "forty" — transformed segment, cursors held
        smap.advance(3)   # "two"
        smap.advance(7)   # "dollars"
        smap.advance(3)   # "and"
        smap.advance(5)   # "fifty"
        smap.advance(5)   # "cents" — segment completes, cursors jump
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

    def advance(self, n_alnum: int) -> None:
        """Consume *n_alnum* TTS alphanumeric chars, advancing internal cursors.

        For unchanged segments the cursors move proportionally through both original
        and LLM text. For transformed segments the cursors are held until the whole
        segment is consumed, then jump to the end of the original segment.

        Args:
            n_alnum: Number of TTS alphanumeric characters to consume.
        """
        self._last_completed = None
        remaining = n_alnum

        while remaining > 0 and self._seg_idx < len(self._segments):
            seg = self._segments[self._seg_idx]
            available = seg.tts_alnum_count - self._seg_consumed
            consume = min(remaining, available)
            self._seg_consumed += consume
            remaining -= consume

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
        """True when mid-flight inside a transformed segment (not yet complete)."""
        if self._seg_idx >= len(self._segments):
            return False
        seg = self._segments[self._seg_idx]
        return seg.is_transformed and self._seg_consumed > 0

    @property
    def last_completed_segment(self) -> TextSegment | None:
        """The segment completed by the last :meth:`advance` call, or ``None``."""
        return self._last_completed

    def reset(self) -> None:
        """Reset all cursor and consumption state to initial values."""
        self._reset_state()
