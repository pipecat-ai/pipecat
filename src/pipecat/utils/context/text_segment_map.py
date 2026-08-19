#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Keeps a position in three versions of the same text as a TTS speaks it."""

import difflib
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum, auto

from pipecat.utils.text.transforms._alnum_utils import (
    advance_by_alnums,
    fold_for_matching,
    normalize,
    strip_trailing_punctuation,
)
from pipecat.utils.text.transforms._markup_utils import (
    raw_offset_after_clean_chars,
    split_markup_runs,
    strip_complete_markup,
    strip_markup,
)


@dataclass(frozen=True)
class TextSegment:
    """A piece of the utterance, paired with what the TTS was given in its place.

    The map is a list of these, laid end to end over the whole utterance. In most
    of them the two sides are identical; the interesting ones are where a
    transform, a filter or a tag made them differ (see :attr:`is_transformed`).

    Parameters:
        original: The piece as a client displays it.
        tts: The same piece as it was sent to the TTS. Identical to *original*
            unless something rewrote it, and empty if it was dropped entirely.
        original_start: Where the piece starts in the full original text.
        original_end: Where it ends. Cursors jump straight here once a rewritten
            piece is finished, since no position inside one means anything.
    """

    original: str
    tts: str
    original_start: int
    original_end: int

    @property
    def is_transformed(self) -> bool:
        """True when the two sides cannot be walked together, character by character.

        Such a segment is all-or-nothing: cursors into the original text wait at
        its start until every spoken word of it has arrived, then jump to its end.

        Any of three things makes it so:

        - the letters and digits differ (``"$42.50"`` against ``"forty two..."``);
        - the two sides split into different numbers of words;
        - the TTS side carries markup, even where the spoken words match. The raw
          cursor has to travel through the tag characters while the original
          cursor has nothing to travel through.

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
    """Where an incoming word sits relative to the segment being offered it.

    A "hop" is one attempt to place a word in one segment. Two outcomes end the
    walk, and two send it on to the next segment.
    """

    PLACED = auto()
    """The word fits here. Advance to the end of the match and stop."""

    CROSSES = auto()
    """The segment only covers the start of the word. Finish the segment and
    carry the unmatched rest to the next one."""

    EXHAUSTED = auto()
    """Nothing speakable is left here, so no word will ever match it -- an empty
    diff side, or a lone ``<break/>``. Finish it and retry the whole word next."""

    NO_MATCH = auto()
    """The word does not belong here at all. Step past any leading punctuation
    and stop, leaving the semantic cursors alone."""


@dataclass(frozen=True)
class _Hop:
    """The outcome of offering one word to one segment.

    Produced by :meth:`TextSegmentMap._classify_hop`, collected by
    :meth:`TextSegmentMap._plan_hops`, and acted on by
    :meth:`TextSegmentMap._consume_word`.

    Parameters:
        kind: Which of the four outcomes applies.
        segment_chars: How far into this segment to move. The matched span for
            ``PLACED``, or a nudge past leading punctuation for ``NO_MATCH``. The
            two draining outcomes leave it 0, since they consume the segment whole.
        word_chars: How much of the word this segment accounted for, and so how
            much to drop before offering the rest to the next segment. Only
            ``CROSSES`` sets it; ``EXHAUSTED`` passes the word on untouched.
    """

    kind: _HopKind
    segment_chars: int = 0
    word_chars: int = 0


class TextSegmentMap:
    """Answers "where are we?" in three versions of one utterance, word by word.

    A TTS provider reports the words it speaks. Each report has to be turned into
    a position -- but into a position in *three* different strings, because the
    same utterance exists in three forms at once:

    - ``tts_text`` -- what was actually spoken, tags and all:
      ``"Your balance is forty two dollars"``
    - ``original_text`` -- what a client displays: ``"Your balance is $42.50"``
    - ``llm_text`` -- what the LLM wrote, so what the transcript should keep:
      ``"Your balance is <b>$42.50</b>"``. Defaults to ``original_text``.

    For a frame nothing rewrote, all three are the same string and every position
    is the same.

    **The hard part is that a spoken word need not appear in the other two.** The
    provider says ``"dollars"``; nothing in ``"$42.50"`` matches it. So the map is
    built once, by diffing ``tts_text`` against ``original_text`` into aligned
    :class:`TextSegment` pieces -- each either survived unchanged or was rewritten
    whole.

    From then on one real cursor moves: ``raw_pos``, how far into ``tts_text`` the
    provider has got. ``user_facing_pos`` and ``llm_pos`` follow it:

    - Through an **unchanged** segment they keep pace, word for word.
    - Through a **rewritten** one they wait. There is no honest position halfway
      through ``"$42.50"`` while ``"forty two dollars"`` is being spoken, so they
      hold and then jump to the end of the span in one step when the last of its
      words lands.

    Callers ask two things. :meth:`word_belongs_current_segment` -- does this token
    plausibly continue what is left to speak? -- and :meth:`advance_word`, which
    consumes it. Both tolerate the ways providers mangle tokens (added punctuation,
    changed case or diacritics, a fragment of a half-open SSML tag) without the
    caller knowing anything about it; :meth:`_classify_hop` holds that logic.

    Example::

        # "$42.50" was sent to the TTS as "forty two dollars and fifty cents"
        smap = TextSegmentMap(
            "Your balance is forty two dollars and fifty cents",
            "Your balance is $42.50",
        )
        for word in ["Your", "balance", "is"]:
            smap.advance_word(word)   # unchanged: every cursor keeps pace
        for word in ["forty", "two", "dollars", "and", "fifty"]:
            smap.advance_word(word)   # rewritten: the other two cursors wait
        smap.advance_word("cents")    # the span is done, so they jump to its end
        assert smap.last_completed_segment.original == "$42.50"
        assert not smap.in_transformed_segment
    """

    def __init__(
        self,
        tts_text: str,
        original_text: str,
        llm_text: str | None = None,
    ):
        """Build the alignment between the three texts.

        The diff happens once, here; everything after this is cursor movement.

        Args:
            tts_text: What was sent to the TTS, and so what incoming words are
                matched against. May carry synthesis tags and rewritten values.
            original_text: The same content as a client displays it, before any
                rewriting. Diffed against *tts_text* to build the segments.
            llm_text: The same content as the LLM wrote it, which may add
                delimiters the other two never see. Rides its own cursor rather
                than being diffed. Defaults to *original_text*.
        """
        self._tts_text = tts_text
        self._original_text = original_text
        self._llm_text = llm_text if llm_text is not None else original_text
        self._segments: list[TextSegment] = self._build(tts_text, original_text)
        self._reset_state()

    @staticmethod
    def _build(tts_text: str, original_text: str) -> list[TextSegment]:
        """Diff the two texts into the list of segments the map walks.

        ``difflib`` compares them a word at a time (whitespace is kept as its own
        token so offsets stay exact) and reports each piece as equal, replaced,
        inserted or deleted. Every piece becomes one :class:`TextSegment`.

        The one refinement: an ``equal`` piece is cut around any markup inside it,
        so a single tag does not make the whole sentence all-or-nothing.

        Called once, from ``__init__``.
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

            # A segment is atomic as soon as it holds any markup, so a tag sitting
            # in the middle of otherwise identical text would freeze the cursors
            # for the whole opcode. Splitting the tag into its own segment keeps
            # that cost to the words the tag actually wraps:
            #
            #     original = tts = "I love to count <spell>1234</spell>."
            #
            #     "I love to count "       plain, cursors advance word by word
            #     "<spell>1234</spell>."   atomic, commits when its last word lands
            #
            # Only "equal" opcodes can be split: both sides hold the same text, so
            # one offset cuts both. The other kinds differ side to side, leaving no
            # shared offset to cut at.
            parts = (
                [(part, part) for part in split_markup_runs(orig_chunk)]
                if tag == "equal"
                else [(orig_chunk, tts_chunk)]
            )

            for orig_part, tts_part in parts:
                orig_end = orig_pos + len(orig_part)
                segments.append(
                    TextSegment(
                        original=orig_part,
                        tts=tts_part,
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
        self._last_leading_duplicate: int = 0

    @staticmethod
    def _prefix_hop(
        candidates: list[tuple[str, int]],
        remaining_word: str,
        require_word_boundary: bool = False,
    ) -> "_Hop | None":
        """Try PLACED/CROSSES of *remaining_word* against *candidates*, literally.

        Shared by the raw and folded passes in :meth:`_classify_hop`
        -- both compare the same way, just on different (length-preserving)
        transforms of the text, so the offsets this returns are valid for
        whichever text produced *candidates* and *remaining_word*.

        Tries an as-is match first, then retries with the word's own trailing
        punctuation removed (some TTS providers add terminal punctuation the
        original text doesn't have, e.g. reading a list item -- ``"my
        account"`` -- as its own sentence, ``"account."``). The segment's own
        punctuation is untouched either way and is still picked up verbatim by
        the next word.

        Args:
            candidates: ``(text, offset)`` pairs to match *remaining_word*
                against, tried in order.
            remaining_word: The word (or its trailing-punctuation-stripped
                variant) to match.
            require_word_boundary: When True, a ``PLACED`` match is only
                accepted if it ends at a word boundary in the candidate (the
                candidate is fully consumed, or the next character is not
                alphanumeric) -- rejecting a short word that only happens to
                be a prefix of a longer one (e.g. ``"account"`` inside
                ``"Accountant"``). Used by the folded pass, where
                folding can otherwise turn a same-case mismatch into a
                spurious mid-word prefix match; the raw pass leaves this off
                to preserve its existing case-sensitive matching.

        Returns:
            A ``PLACED`` or ``CROSSES`` hop, or ``None`` if nothing matched.
        """
        trimmed_word = strip_trailing_punctuation(remaining_word)
        words = (
            (remaining_word,)
            if trimmed_word == remaining_word
            else (
                remaining_word,
                trimmed_word,
            )
        )
        for word in words:
            if not word:
                continue
            for candidate, offset in candidates:
                if candidate.startswith(word):
                    lands_mid_word = (
                        require_word_boundary
                        and len(word) < len(candidate)
                        and candidate[len(word)].isalnum()
                    )
                    if not lands_mid_word:
                        return _Hop(_HopKind.PLACED, segment_chars=offset + len(word))
                elif candidate and word.startswith(candidate):
                    return _Hop(_HopKind.CROSSES, word_chars=len(candidate))
        return None

    @staticmethod
    def _leading_nonalnum_len(text: str, stop_at_markup: bool = False) -> int:
        """Length of *text*'s leading run of non-alphanumeric characters.

        With *stop_at_markup*, the run also stops at a ``'<'``. A tag's letters
        are markup, not the start of spoken content, so a scan that reached into
        one would let a tag *name* arriving as its own word-timestamp token (e.g.
        ``"break"`` against ``"<break/>hello"``) match as if it were spoken,
        consuming the tag and desyncing the cursor.
        """
        i = 0
        while i < len(text) and not text[i].isalnum():
            if stop_at_markup and text[i] == "<":
                break
            i += 1
        return i

    @staticmethod
    def _classify_hop(segment_remaining: str, remaining_word: str) -> _Hop:
        """Decide where *remaining_word* goes against this segment's remaining raw text.

        Purely positional/textual -- no tag-name parsing or cross-call state. The
        word is checked with three matching strategies, in order:

        1. Literal, tried at three progressively deeper skip offsets into the
           segment: as-is (for providers whose word tokens carry their own
           surrounding whitespace, e.g. Inworld's ``" world"``), past leading
           whitespace, and past the whole leading non-alphanumeric run -- the
           last covering punctuation a provider leaves behind by not repeating it
           in its word-timestamp events (e.g. the ``", "`` still pending in
           ``"Yeah, I can"`` when ``"I"`` arrives). The word's own trailing
           punctuation is also tried removed (see :meth:`_prefix_hop`).
        2. Same as 1, with both sides folded by
           :func:`~pipecat.utils.text.transforms._alnum_utils.fold_for_matching`:
           for a provider that lowercases a word, strips its diacritics, or
           normalizes typographic punctuation in word-timestamp events (e.g.
           ``"SQL"`` -> ``"sql"``, ``"café"`` -> ``"cafe"``, ``"don’t"`` ->
           ``"don't"``).
           Folding is a length-preserving, per-character transform (unlike
           :func:`normalize`, it never drops or merges characters), so an
           offset found against the folded text applies unchanged to the
           original. Folding erases case, which could otherwise turn a short
           word into a spurious mid-word prefix match against a longer one
           (e.g. folded ``"account"`` against ``"Accountant"``); a
           ``PLACED`` match is only accepted here if it lands on a word
           boundary (see :meth:`_prefix_hop`'s ``require_word_boundary``).
        3. Markup-stripped on both sides: for a provider that wraps the word
           token in tags absent from ``tts_text`` (or vice versa). Recomputed
           fresh each call -- no persisted tag state. As in 1 and 2, the word's
           own trailing punctuation is also tried removed: a provider may end a
           tagged span with punctuation the source text left to a following line
           break.

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
          substitution): the word doesn't belong here, so ``segment_chars`` carries a
          nudge past the segment's leading run of non-alphanumeric chars only --
          never past real spoken content.
        """
        stripped = segment_remaining.lstrip()
        lead_ws = len(segment_remaining) - len(stripped)
        lead_nonalnum = TextSegmentMap._leading_nonalnum_len(segment_remaining, stop_at_markup=True)

        # Strategy 1: literal match at progressively deeper skip offsets -- the
        # text as-is, past leading whitespace, then past the whole leading
        # non-alphanumeric run (punctuation the provider didn't repeat as its own
        # token, e.g. the ", " left in "Yeah, I can" once "Yeah" is consumed).
        # The offsets are non-decreasing, so this tries the least aggressive skip
        # first; duplicates are dropped.
        candidates = [
            (segment_remaining[offset:], offset)
            for offset in dict.fromkeys((0, lead_ws, lead_nonalnum))
        ]
        hop = TextSegmentMap._prefix_hop(candidates, remaining_word)
        if hop is not None:
            return hop

        # Strategy 2: same candidates, variation-folded. require_word_boundary
        # guards against folding turning a short word into a false mid-word
        # prefix match against a longer one that only differs in case.
        folded_word = fold_for_matching(remaining_word)
        folded_candidates = [(fold_for_matching(c), offset) for c, offset in candidates]
        hop = TextSegmentMap._prefix_hop(folded_candidates, folded_word, require_word_boundary=True)
        if hop is not None:
            return hop

        # Strategy 3: markup-stripped match.
        haystack = strip_markup(stripped)
        clean_word = strip_markup(remaining_word)
        trimmed_word = strip_trailing_punctuation(clean_word)
        clean_words = (clean_word,) if trimmed_word == clean_word else (clean_word, trimmed_word)
        for candidate in clean_words:
            if candidate and haystack.startswith(candidate):
                raw_len = raw_offset_after_clean_chars(stripped, len(candidate))
                return _Hop(_HopKind.PLACED, segment_chars=lead_ws + raw_len)

        # Nothing spoken left here: drain so the word can try the next segment.
        if not normalize(segment_remaining):
            return _Hop(_HopKind.EXHAUSTED)

        # Foreign token: nudge past leading punctuation only, then stop. Unlike
        # the strategy 1 candidates this does not stop at markup -- it moves the
        # raw cursor rather than deciding a match, so there is no tag name it
        # could mistake for spoken content.
        return _Hop(
            _HopKind.NO_MATCH, segment_chars=TextSegmentMap._leading_nonalnum_len(segment_remaining)
        )

    def _advance_cursors_to(self, seg: TextSegment, new_pos: int) -> None:
        """Move every cursor to *new_pos* within *seg*, and finish *seg* if reached.

        This is where the "keep pace or wait" rule from the class docstring is
        actually applied, and the only place the two derived cursors move.

        **Unchanged segment** -- both sides hold the same text, so the derived
        cursors keep pace. They are spent an *alphanumeric budget*: however many
        letters and digits this step consumed on the TTS side is how many they
        advance by. :func:`advance_by_alnums` spends it, stepping over markup for
        free (which is how a tag joins the word next to it) and sweeping up
        punctuation trailing the word (``"you?"`` moves as one).

        **Rewritten segment** -- the derived cursors do not move at all, until the
        segment is finished; then they jump to the end of its span.

        Either way the cursor stops short of trailing whitespace, which belongs to
        the token that follows.
        """
        if seg.is_transformed:
            # Whatever is left is only a closing tag or the like, which no word
            # event will ever name. Take it now so the segment can finish.
            # Unchanged segments are not given this: a trailing emoji there is
            # real output, and its own event is still coming.
            if not normalize(seg.tts[new_pos:]):
                new_pos = len(seg.tts)
        else:
            n_alnum = len(normalize(seg.tts[self._seg_raw_pos : new_pos]))
            if n_alnum:
                self._user_facing_pos = advance_by_alnums(
                    self._original_text, self._user_facing_pos, n_alnum
                )
            else:
                # A token with no letters or digits to spend -- punctuation set
                # off by a space, as French writes it ("va ?", "Attention :").
                # There is no budget to advance by, so step straight to where the
                # raw cursor got to, and the mark leaves the remaining text now
                # rather than a word later. Both sides are identical here, so that
                # offset is exact.
                self._user_facing_pos = seg.original_start + len(seg.tts[:new_pos].rstrip())
            self._llm_pos = advance_by_alnums(self._llm_text, self._llm_pos, n_alnum)

        self._seg_raw_pos = new_pos

        # Reached the end of the segment: hand the derived cursors the jump they
        # have been waiting for, and move on to the next segment.
        if new_pos >= len(seg.tts):
            if seg.is_transformed:
                self._user_facing_pos = seg.original_end
                self._llm_pos = advance_by_alnums(
                    self._llm_text, self._llm_pos, seg.original_alnum_count
                )
            self._last_completed = seg
            self._seg_idx += 1
            self._seg_raw_pos = 0

    def _plan_hops(self, word: str) -> tuple[list[_Hop], str]:
        """Offer *word* to each segment from the cursor on, changing nothing.

        The decision half of consuming a word, kept separate from the cursor
        movement so that :meth:`_consume_word` and :meth:`_can_consume_word`
        cannot drift apart: a token one of them accepts is by construction a
        token the other places.

        Most words are placed by the first segment tried and the walk stops
        there. It continues when a segment cannot finish the job -- because the
        word outruns it, or because it has nothing speakable to offer -- in which
        case whatever is left of the word moves on to the next segment.

        Returns:
            The hop each segment produced, in walk order, and whatever is left of
            the word after the last segment. A non-empty remainder is the word
            running past the end of this frame.
        """
        seg_idx = self._seg_idx
        raw_pos = self._seg_raw_pos
        remaining_word = word
        hops: list[_Hop] = []

        while remaining_word and seg_idx < len(self._segments):
            hop = self._classify_hop(self._segments[seg_idx].tts[raw_pos:], remaining_word)
            hops.append(hop)

            # PLACED and NO_MATCH both end the walk; the other two carry on.
            if hop.kind is _HopKind.PLACED or hop.kind is _HopKind.NO_MATCH:
                return hops, ""

            remaining_word = remaining_word[hop.word_chars :]
            seg_idx += 1
            raw_pos = 0

        return hops, remaining_word

    def _consume_word(self, word: str) -> None:
        """Apply the hops *word* takes, moving the cursors as each one says.

        Anything the walk could not place is the word running past the end of
        this frame, and is left in ``last_overflow`` for the caller to give to
        the next frame.
        """
        hops, overflow = self._plan_hops(word)

        for hop in hops:
            seg = self._segments[self._seg_idx]

            if hop.kind is _HopKind.NO_MATCH:
                # The word belongs somewhere else entirely (a provider swapping a
                # symbol, say). Nudge the raw cursor past any leading punctuation
                # so the next word is not blocked by it, but leave the cursors
                # that mean something alone -- nothing was really spoken here.
                self._seg_raw_pos += hop.segment_chars
            elif hop.kind is _HopKind.PLACED:
                self._advance_cursors_to(seg, self._seg_raw_pos + hop.segment_chars)
            else:
                # CROSSES or EXHAUSTED: this segment is done either way, and the
                # next hop was classified against the one after it.
                self._advance_cursors_to(seg, len(seg.tts))

        if overflow:
            self._last_overflow = overflow

    def advance_word(self, word: str) -> None:
        """Consume one spoken word, moving every cursor to where it leaves off.

        Afterwards :attr:`last_completed_segment`, :attr:`last_overflow` and
        :attr:`last_leading_duplicate` describe what this particular word did;
        each is cleared at the start of the next call.

        Args:
            word: One token from the provider's word-timestamp stream. It may be
                a plain word, a word carrying its own spacing or punctuation, or a
                fragment of a half-open tag -- matching is textual, so the caller
                does not have to know which.
        """
        self._last_completed = None
        self._last_overflow = None
        self._last_leading_duplicate = 0

        if word:
            self._last_leading_duplicate = self._leading_duplicate_len(word)
            self._consume_word(word)

    def _leading_duplicate_len(self, word: str) -> int:
        """How many leading chars of *word* repeat punctuation already consumed.

        In ``"Yeah, I can"`` the comma is swept into ``"Yeah"``'s span, so a
        provider reporting the next token as ``", I"`` rather than ``"I"``
        carries it twice; this returns 2, the comma and its space.

        Returns 0 when that leading punctuation is new content instead
        (``'"hello'``), and when the token is punctuation only -- the mark
        arriving as its own event. Call before advancing, while ``llm_pos``
        still ends the previous word's span.
        """
        i = 0
        while i < len(word) and (
            word[i].isspace() or unicodedata.category(word[i]).startswith("P")
        ):
            i += 1
        if i >= len(word):
            return 0
        mark = word[:i].strip()
        if not mark:
            return 0
        start = self._llm_pos - len(mark)
        if start < 0 or self._llm_text[start : self._llm_pos] != mark:
            return 0
        return i

    def word_belongs_current_segment(self, word: str) -> bool:
        """Return True if *word* could plausibly be the next thing spoken here.

        :meth:`advance_word` without the advancing, so callers can look before
        they leap. A False answer means the provider skipped something, and the
        caller should give this word to the next frame instead.

        A word with no letters or digits gets a second chance through
        :meth:`_symbol_belongs_here`, since there is nothing in it to match on.
        """
        if not word:
            return True
        if self._can_consume_word(word):
            return True
        if not normalize(word):
            return self._symbol_belongs_here(word)
        return False

    def _can_consume_word(self, word: str) -> bool:
        """Answer *would this word be placed?* without placing it.

        True once some segment would place the word, or once it has run through
        every remaining segment. False if there is nothing left to offer, or a
        segment rejects the word outright.
        """
        if self._seg_idx >= len(self._segments):
            return False

        hops, _ = self._plan_hops(word)
        return not hops or hops[-1].kind is not _HopKind.NO_MATCH

    def _symbol_belongs_here(self, word: str) -> bool:
        """Return True if a word of pure punctuation or symbols belongs here.

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
    def last_leading_duplicate(self) -> int:
        """Leading chars of the last :meth:`advance_word` token already consumed.

        The mirror of :attr:`last_overflow`: that reports the token's tail
        running past this text, this its head repeating the previous word's
        punctuation. Slice off both to get the token's share of this frame::

            word[last_leading_duplicate : len(word) - len(last_overflow or "")]
        """
        return self._last_leading_duplicate

    @property
    def is_complete(self) -> bool:
        """True once every segment's alphanumeric content has been accounted for.

        Not simply "cursor past the last segment": a frame whose remaining
        content is entirely punctuation/markup (zero alphanumeric chars) is
        already complete even if its raw text hasn't been walked yet -- with one
        exception. Trailing punctuation separated from the preceding word by
        whitespace (e.g. the ``?`` in the French ``"Comment ça va ?"``) is
        emitted by the TTS as its own word-timestamp token, so the frame is not
        complete until that token arrives (see :meth:`_pending_separated_punctuation`).
        Punctuation attached directly to the last word (``"you?"``) was already
        consumed with it, and trailing markup (closing tags) never arrives as its
        own token, so neither holds completion open.
        """
        if self._seg_idx >= len(self._segments):
            return True
        seg = self._segments[self._seg_idx]
        if normalize(seg.tts[self._seg_raw_pos :]):
            return False
        if self._pending_separated_punctuation(seg.tts[self._seg_raw_pos :]):
            return False
        return all(not normalize(s.tts) for s in self._segments[self._seg_idx + 1 :])

    @staticmethod
    def _pending_separated_punctuation(remaining: str) -> bool:
        """True when *remaining* is a whitespace-separated trailing punctuation token.

        The TTS emits punctuation set off from its word by a space (French/other
        locales: ``"va ?"``, ``"Bonjour !"``) as its own word-timestamp event, so
        the segment must stay open until it arrives. Restricted to Unicode
        punctuation (category ``P*``): a trailing emoji or symbol (e.g. ``"day! 😊"``,
        ``"→"``) is never spoken as its own token, so it must not hold completion
        open. Markup-only remainders (a closing tag) are stripped first since they
        never arrive as their own token either. Called only once no alphanumeric
        content remains.
        """
        stripped_markup = strip_complete_markup(remaining)
        if not stripped_markup[:1].isspace():
            return False
        content = stripped_markup.strip()
        return bool(content) and unicodedata.category(content[0]).startswith("P")

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
