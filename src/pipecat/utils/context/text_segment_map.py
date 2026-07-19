#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Segment-level mapping between original and TTS-transformed text."""

import difflib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto

from pipecat.utils.text.transforms._alnum_utils import advance_by_alnums, normalize


def _iter_clean_chars(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(raw_index, char)`` for each character of *text* outside markup.

    The single definition of "what is markup" -- anything between '<' and '>',
    syntax-based and tag-name independent -- shared by :func:`strip_markup` and
    :func:`_raw_len_for_clean_chars` so the two can't disagree. An unclosed '<'
    swallows the rest of the string (matching how word-timestamp fragments can
    arrive mid-tag).
    """
    in_tag = False
    for i, ch in enumerate(text):
        if in_tag:
            if ch == ">":
                in_tag = False
        elif ch == "<":
            in_tag = True
        else:
            yield i, ch


def strip_markup(text: str) -> str:
    """Remove XML/SSML-like markup from a word-timestamp fragment.

    Syntax-based, not tag-name based: treats anything between '<' and '>' as
    markup and preserves text outside it. An unclosed '<' swallows the rest of
    *text*, matching how a raw word-timestamp token can arrive mid-tag (see
    :func:`_iter_clean_chars`).

    For a *complete* text (not a possibly-truncated fragment), use
    :func:`strip_complete_markup` instead -- swallowing the rest of the string
    past a lone '<' is only correct for a genuinely truncated tag; in a
    complete text a lone '<' is real content (e.g. ``"5 < 10"`` or ``"<3"``).

    Used by :meth:`TextSegmentMap._classify_hop`'s markup-stripped matching,
    where the incoming word may be a fragment of a still-open tag.
    """
    return "".join(ch for _, ch in _iter_clean_chars(text))


def strip_complete_markup(text: str) -> str:
    """Remove well-formed '<...>' markup from a complete, static text.

    Unlike :func:`strip_markup`, only strips matched '<...>' pairs -- a lone
    '<' with no later '>' is left in place as real content rather than
    swallowing the rest of *text*, since there is no streamed fragment here
    that could be mid-tag. Mirrors the tag-stripping regex in
    :func:`pipecat.utils.text.transforms._alnum_utils.normalize`.

    Used by :attr:`TextSegment.is_transformed` and by
    :class:`~pipecat.utils.context.word_completion_tracker.WordCompletionTracker`
    to default ``user_facing_text`` to a tag-free string.
    """
    return re.sub(r"<[^>]+>", "", text)


def _raw_len_for_clean_chars(text: str, n: int) -> int:
    """Return the raw offset into *text* just past its *n*-th markup-stripped char.

    Inverse of :func:`strip_markup` for a prefix: where ``strip_markup`` collects
    every non-markup char, this finds the raw index one past the *n*-th of them --
    converting a match measured in markup-stripped space back to a raw offset.
    Returns ``len(text)`` when *text* has fewer than *n* non-markup chars.
    """
    if n <= 0:
        return 0
    seen = 0
    for i, _ in _iter_clean_chars(text):
        seen += 1
        if seen == n:
            return i + 1
    return len(text)


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

        This holds when:

        - alphanumeric content differs between original and TTS sides;
        - a replacement changed word count / tokenization;
        - the TTS side contains markup, even if the spoken alphanumeric content is
          the same as the original.

        The markup check is syntax-based and tag-name independent. For example,
        ``<phoneme ...>Siobhan</phoneme>`` is transformed because the TTS segment
        has raw markup around the original word, so the raw segment cursor can move
        while the original/LLM cursors must remain held.
        """
        if self.tts != strip_complete_markup(self.tts):
            return True
        if normalize(self.original) != normalize(self.tts):
            return True
        return len(self.original.split()) != len(self.tts.split())

    @property
    def tts_alnum_count(self) -> int:
        """Number of alphanumeric characters in the spoken TTS content."""
        return len(normalize(self.tts))

    @property
    def original_alnum_count(self) -> int:
        """Number of alphanumeric characters in the original side of this segment."""
        return len(normalize(self.original))


class _HopKind(Enum):
    """How an incoming word relates to the current segment's remaining raw text."""

    PLACED = auto()  # word fits within this segment; stop here
    CROSSES = auto()  # word runs past this segment; drain it and carry the remainder
    EXHAUSTED = auto()  # no spoken content left here; drain it, keep the whole word
    NO_MATCH = auto()  # word doesn't belong here; nudge past leading punctuation, stop


@dataclass(frozen=True)
class _Hop:
    """Result of matching one word against one segment.

    Produced by :meth:`TextSegmentMap._classify_hop` and consumed by the segment
    walk in :meth:`TextSegmentMap._advance_raw` (and its read-only twin
    :meth:`TextSegmentMap._word_matches_remaining`).

    Parameters:
        kind: Which relationship holds.
        seg_chars: Raw chars consumed within this segment -- the matched span for
            ``PLACED``, the leading non-alphanumeric nudge for ``NO_MATCH``; 0 for
            the draining kinds (``CROSSES``/``EXHAUSTED``), which always drain the
            whole segment.
        word_chars: Chars trimmed off the front of the word before continuing to
            the next segment. Meaningful for ``CROSSES``; 0 otherwise.
    """

    kind: _HopKind
    seg_chars: int = 0
    word_chars: int = 0


class TextSegmentMap:
    """Maps cursor positions across three parallel texts as TTS words stream in.

    The three texts describe the same utterance at different stages:

    - ``tts_text``: what was sent to the TTS service (may carry SSML markup and
      text transforms, e.g. ``"forty two dollars and fifty cents"``).
    - ``original_text``: the user-facing string (no markup/transforms, e.g.
      ``"$42.50"``).
    - ``llm_text``: the LLM-produced string, which may add delimiters (e.g.
      ``<card>$42.50</card>``); defaults to ``original_text``.

    Built once by diffing ``tts_text`` against ``original_text`` into aligned
    :class:`TextSegment` chunks. A single cursor drives everything -- ``raw_pos``,
    the position reached in ``tts_text`` as words are consumed. The
    ``user_facing_pos`` and ``llm_pos`` cursors are derived from it:

    - Across an **unchanged** segment they advance proportionally, char for char.
    - Across a **transformed** segment (alnum content, tokenization, or markup
      differs) they are held until the segment's entire raw text is consumed,
      then jump to the end of its original span in one step -- the transform is
      atomic, so there is no meaningful mid-segment original position.

    Callers drive the map word-by-word: :meth:`word_belongs_current_segment`
    asks whether a raw word-timestamp token plausibly continues the remaining
    TTS text, and :meth:`advance_word` consumes it. Both match the token against
    the segment's remaining raw text directly -- literally, or (as a stateless
    fallback) with markup stripped from both sides -- so a token that is a
    fragment of a still-open SSML tag (e.g. an attribute-only word from a
    multi-attribute tag some TTS providers split across several word-timestamp
    events) needs no special tag parsing.

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
            original_text: User-facing pre-transform text.
            llm_text: LLM-produced text, which may have surrounding tags. Defaults
                to ``original_text`` when not provided.
        """
        self._tts_text = tts_text
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
        self._seg_raw_pos: int = 0
        self._user_facing_pos: int = 0
        self._llm_pos: int = 0
        self._last_completed: TextSegment | None = None
        self._last_overflow: str | None = None

    @staticmethod
    def _classify_hop(segment_remaining: str, remaining_word: str) -> _Hop:
        """Decide where *remaining_word* goes against this segment's remaining raw text.

        Purely positional/textual -- no tag-name parsing or cross-call state. The
        word is checked with three matching strategies, in order:

        1. Literal, as-is: for providers whose word tokens carry their own
           surrounding whitespace (e.g. Inworld's ``" world"``).
        2. Literal, with the segment's leading whitespace stripped: the common
           case where the word omits the separating space.
        3. Markup-stripped on both sides: for a provider that wraps the word
           token in tags absent from ``tts_text`` (or vice versa). Recomputed
           fresh each call -- no persisted tag state.

        Strategies 1 and 2 yield :attr:`_HopKind.PLACED` (word fits inside this
        segment) or :attr:`_HopKind.CROSSES` (the segment's remaining text is
        only a prefix of the word, which spills into the next segment). Strategy
        3 only yields ``PLACED``.

        If none match, the outcome is structural:

        - :attr:`_HopKind.EXHAUSTED` when no alphanumeric content is left to
          speak here (a self-closing ``<break/>`` tag, or only trailing
          whitespace/punctuation): drain the segment so the word can try the
          next one. Checked only after the match attempts, so a word that *does*
          literally match trailing non-alnum content (e.g. an emoji) is still
          found here rather than skipped over.
        - :attr:`_HopKind.NO_MATCH` otherwise (e.g. a provider symbol
          substitution): the word doesn't belong here, so ``seg_chars`` carries a
          nudge past the segment's leading run of non-alphanumeric chars only --
          never past real spoken content.
        """
        stripped = segment_remaining.lstrip()
        lead_ws = len(segment_remaining) - len(stripped)

        # Strategies 1 and 2: literal match, as-is then whitespace-stripped.
        candidates = [(segment_remaining, 0)]
        if lead_ws:
            candidates.append((stripped, lead_ws))
        for candidate, offset in candidates:
            if candidate.startswith(remaining_word):
                return _Hop(_HopKind.PLACED, seg_chars=offset + len(remaining_word))
            if candidate and remaining_word.startswith(candidate):
                return _Hop(_HopKind.CROSSES, word_chars=len(candidate))

        # Strategy 3: markup-stripped match.
        clean_word = strip_markup(remaining_word)
        if clean_word and strip_markup(stripped).startswith(clean_word):
            raw_len = _raw_len_for_clean_chars(stripped, len(clean_word))
            return _Hop(_HopKind.PLACED, seg_chars=lead_ws + raw_len)

        # Nothing spoken left here: drain so the word can try the next segment.
        if not normalize(segment_remaining):
            return _Hop(_HopKind.EXHAUSTED)

        # Foreign token: nudge past leading punctuation only, then stop.
        nudge = 0
        while nudge < len(segment_remaining) and not segment_remaining[nudge].isalnum():
            nudge += 1
        return _Hop(_HopKind.NO_MATCH, seg_chars=nudge)

    def _commit_raw_span(self, seg: TextSegment, new_pos: int) -> None:
        """Advance the raw cursor to *new_pos* in *seg*, moving the semantic cursors.

        For an unchanged segment, ``user_facing_pos``/``llm_pos`` advance
        proportionally to the alphanumeric content just consumed -- never snapped
        to ``original_end`` (which would overshoot a segment ending in trailing
        whitespace). A transformed segment holds those cursors until it fully
        completes, then jumps them to the end of its original span in one step.
        """
        if seg.is_transformed:
            # A trailing markup-only remainder (e.g. a closing tag) never arrives
            # as its own word-timestamp event, so once no spoken content is left
            # after *new_pos*, fold it in and let the segment complete. (Unchanged
            # segments don't get this: a trailing symbol/emoji there is a real
            # output position that IS expected to arrive as its own word.)
            if not normalize(seg.tts[new_pos:]):
                new_pos = len(seg.tts)
        else:
            n_alnum = len(normalize(seg.tts[self._seg_raw_pos : new_pos]))
            self._user_facing_pos = advance_by_alnums(
                self._original_text, self._user_facing_pos, n_alnum
            )
            self._llm_pos = advance_by_alnums(self._llm_text, self._llm_pos, n_alnum)

        self._seg_raw_pos = new_pos

        if new_pos >= len(seg.tts):
            if seg.is_transformed:
                self._user_facing_pos = seg.original_end
                self._llm_pos = advance_by_alnums(
                    self._llm_text, self._llm_pos, seg.original_alnum_count
                )
            self._last_completed = seg
            self._seg_idx += 1
            self._seg_raw_pos = 0

    def _advance_raw(self, word: str) -> None:
        """Match *word* against the remaining raw TTS text, advancing cursors.

        Hops across segments as needed for a word that straddles a segment
        boundary. If the word runs past the end of ``tts_text`` (no segments
        left to carry the remainder into), the unconsumed raw suffix is stored
        in ``last_overflow``.
        """
        remaining_word = word

        while remaining_word and self._seg_idx < len(self._segments):
            seg = self._segments[self._seg_idx]
            old_pos = self._seg_raw_pos
            hop = self._classify_hop(seg.tts[old_pos:], remaining_word)

            if hop.kind is _HopKind.NO_MATCH:
                # Foreign token (e.g. a provider symbol substitution): move the
                # raw cursor past the leading punctuation only -- never the
                # semantic cursors -- and stop.
                self._seg_raw_pos = old_pos + hop.seg_chars
                return

            if hop.kind is _HopKind.PLACED:
                # Word sits inside this segment; advance to the matched end and stop.
                self._commit_raw_span(seg, old_pos + hop.seg_chars)
                return

            # CROSSES or EXHAUSTED: drain the whole segment and carry whatever
            # part of the word it didn't account for into the next one.
            self._commit_raw_span(seg, len(seg.tts))
            remaining_word = remaining_word[hop.word_chars :]

        if remaining_word:
            self._last_overflow = remaining_word

    def advance_word(self, word: str) -> None:
        """Match *word* against the remaining TTS text and advance cursors.

        Args:
            word: Raw TTS word-timestamp token. May be a fragment of a tag, a
                spoken word, or a mix -- the matching is purely textual, no
                tag parsing is required from callers.
        """
        self._last_completed = None
        self._last_overflow = None

        if word:
            self._advance_raw(word)

    def word_belongs_current_segment(self, word: str) -> bool:
        """Return True if *word* plausibly continues the remaining TTS text.

        A non-mutating dry run of the same matching :meth:`advance_word` uses.
        Used to detect when a TTS provider silently dropped a word-timestamp
        event: if the incoming word does not match, the caller should
        force-complete this slot and route the word to the next.
        """
        if not word:
            return True
        if self._word_matches_remaining(word):
            return True
        if not normalize(word):
            return self._symbol_word_belongs(word)
        return False

    def _word_matches_remaining(self, word: str) -> bool:
        """Read-only replay of :meth:`_advance_raw`'s segment walk; mutates nothing.

        Returns True once the word is PLACED (fits within the current segment, or
        CROSSES through fully-drained segments into one that places it), or once a
        straddle drains every remaining segment. Returns False when the map is
        already exhausted, or a hop is NO_MATCH (no recognizable match at all).
        """
        if self._seg_idx >= len(self._segments):
            return False

        seg_idx = self._seg_idx
        raw_pos = self._seg_raw_pos
        remaining_word = word

        while remaining_word and seg_idx < len(self._segments):
            hop = self._classify_hop(self._segments[seg_idx].tts[raw_pos:], remaining_word)

            if hop.kind is _HopKind.PLACED:
                return True
            if hop.kind is _HopKind.NO_MATCH:
                return False

            # CROSSES or EXHAUSTED: keep hopping into the next segment.
            remaining_word = remaining_word[hop.word_chars :]
            seg_idx += 1
            raw_pos = 0

        return True

    def _symbol_word_belongs(self, word: str) -> bool:
        """Return True if a non-alnum word (emoji, punctuation, symbol) belongs here.

        Two checks are applied in order:

        1. **Literal substring**: search for the raw word in the remaining TTS
           text. The search window is backed up over any already-consumed
           trailing punctuation, since that may have been swept past already.

        2. **Symbol substitution fallback**: some TTS providers substitute
           Unicode symbols with ASCII punctuation in word-timestamp events (e.g.
           ElevenLabs reports "->" as "-"), so check 1 always fails even though
           the word belongs here. If alnum content still remains unconsumed and
           the next non-space character in the TTS text is itself a non-alnum
           symbol, accept the word as a substitution.
        """
        pos = self.raw_pos
        search_start = pos
        while search_start > 0:
            ch = self._tts_text[search_start - 1]
            if ch.isalnum() or ch.isspace() or ch == ">":
                break
            search_start -= 1
        if word in self._tts_text[search_start:]:
            return True

        if self._seg_idx >= len(self._segments):
            return False

        while pos < len(self._tts_text) and self._tts_text[pos].isspace():
            pos += 1
        return pos < len(self._tts_text) and not self._tts_text[pos].isalnum()

    @property
    def user_facing_pos(self) -> int:
        """Current byte offset in the original user-facing text."""
        return self._user_facing_pos

    @property
    def llm_pos(self) -> int:
        """Current byte offset in the LLM text."""
        return self._llm_pos

    @property
    def raw_pos(self) -> int:
        """Current global byte offset into ``tts_text``."""
        pos = sum(len(s.tts) for s in self._segments[: self._seg_idx])
        if self._seg_idx < len(self._segments):
            pos += self._seg_raw_pos
        return pos

    @property
    def last_overflow(self) -> str | None:
        """Raw suffix of the last :meth:`advance_word` call that overflowed.

        ``None`` unless that call's word ran past the end of ``tts_text`` (no
        segments left to carry the remainder into). Always a suffix of the
        word passed to that call -- the consumed prefix is
        ``word[: len(word) - len(last_overflow)]``.
        """
        return self._last_overflow

    @property
    def is_complete(self) -> bool:
        """True once every segment's alphanumeric content has been accounted for.

        Not simply "cursor past the last segment": a frame whose remaining
        content is entirely punctuation/markup (zero alphanumeric chars) is
        already complete even if its raw text hasn't been walked yet.
        """
        if self._seg_idx >= len(self._segments):
            return True
        seg = self._segments[self._seg_idx]
        if normalize(seg.tts[self._seg_raw_pos :]):
            return False
        return all(not normalize(s.tts) for s in self._segments[self._seg_idx + 1 :])

    @property
    def in_transformed_segment(self) -> bool:
        """True when the cursor is on a transformed segment that is not complete."""
        if self._seg_idx >= len(self._segments):
            return False

        seg = self._segments[self._seg_idx]
        return seg.is_transformed and self._seg_raw_pos > 0

    @property
    def last_completed_segment(self) -> TextSegment | None:
        """The segment completed by the last :meth:`advance_word` call."""
        return self._last_completed

    def reset(self) -> None:
        """Reset all cursor and consumption state."""
        self._reset_state()
