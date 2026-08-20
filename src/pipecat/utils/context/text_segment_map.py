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

from pipecat.utils.text.alnum_utils import (
    advance_by_alnums,
    alnum_only,
    fold_for_matching,
    has_alnum,
    strip_trailing_punctuation,
)
from pipecat.utils.text.markup_utils import (
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
        """True when the two sides cannot be followed together, character by character.

        A segment like this is all or nothing. The cursors into the other texts
        wait at its start until every spoken word of it has arrived, then jump
        straight to its end, because no position inside it means anything.

        Any one of these makes it true:

        - the letters and digits differ, as in ``"$42.50"`` against
          ``"forty two dollars"``;
        - the two sides have different numbers of words;
        - the TTS side has tags in it, even when the words match. The cursor
          through the spoken text has tag characters to cross that the other
          texts do not have.

        Only the shape of a tag matters, never its name: ``<phoneme
        ...>Siobhan</phoneme>`` counts because of the tags around the word.
        """
        if self.tts != strip_complete_markup(self.tts):
            return True
        if alnum_only(self.original) != alnum_only(self.tts):
            return True
        return len(self.original.split()) != len(self.tts.split())

    @property
    def tts_alnum_count(self) -> int:
        """How many letters and digits the spoken side of this segment has.

        This is what the provider's words spend as they arrive. On a rewritten
        segment it has nothing to do with :attr:`original_alnum_count`:
        "forty two dollars" against "$42.50".
        """
        return len(alnum_only(self.tts))

    @property
    def original_alnum_count(self) -> int:
        """How many letters and digits the original side of this segment has.

        This is what the cursors into ``original_text`` and ``llm_text`` spend,
        since those texts hold the original characters, not the spoken ones.
        """
        return len(alnum_only(self.original))


class _HopKind(Enum):
    """What happened when a word was offered to one segment.

    Trying one word against one segment is called a hop. Two of these answers
    end the attempt; the other two send the word on to the next segment.
    """

    PLACED = auto()
    """The word fits here. Move to the end of the match and stop."""

    CROSSES = auto()
    """This segment holds only the beginning of the word. Finish the segment and
    take the rest of the word to the next one."""

    EXHAUSTED = auto()
    """There is nothing here that can be spoken, so no word will ever match --
    an empty side of the diff, or a lone ``<break/>``. Finish the segment and
    try the whole word again on the next one."""

    NO_MATCH = auto()
    """The word does not belong here. Step past punctuation at the start of the
    segment and stop, without moving the cursors into the other two texts."""


@dataclass(frozen=True)
class _Hop:
    """The outcome of offering one word to one segment.

    Produced by :meth:`TextSegmentMap._classify_hop`, collected by
    :meth:`TextSegmentMap._plan_hops`, and acted on by
    :meth:`TextSegmentMap._consume_word`.

    Parameters:
        kind: Which of the four outcomes happened.
        segment_advance: How many characters to move forward in this segment.
            The length of the match for ``PLACED``, or a step past leading
            punctuation for ``NO_MATCH``. The other two leave it at 0 because
            they finish the whole segment anyway.
        word_consumed: How many characters of the word this segment used up, and
            so how many to drop before offering the rest to the next segment.
            Only ``CROSSES`` sets it; ``EXHAUSTED`` passes the word on whole.
    """

    kind: _HopKind
    segment_advance: int = 0
    word_consumed: int = 0


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

    ``llm_text`` is never compared against the others, and does not need to be.
    It holds the same letters and digits as ``original_text``, in the same
    order, and differs only in what is wrapped around them -- tags, delimiters,
    punctuation. So counting letters and digits is enough to keep it in step,
    and its cursor moves by that count.

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
        """Line the three texts up against each other.

        The comparison happens once, here. Everything after this only moves
        cursors.

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
        self._segments: list[TextSegment] = self._build_segments(tts_text, original_text)
        self._reset_state()

    @staticmethod
    def _build_segments(tts_text: str, original_text: str) -> list[TextSegment]:
        """Compare the two texts and cut them into the segments the map walks.

        ``difflib`` lines them up a word at a time and reports each piece as
        equal, replaced, inserted or deleted. Every piece becomes one
        :class:`TextSegment`. Spaces are kept as words of their own so that the
        positions stay exact.

        One extra step: a piece that came out equal is cut around any tag inside
        it, so a single tag does not turn the whole sentence into an
        all-or-nothing segment.

        Called once, from ``__init__``.
        """

        def tokenize(text: str) -> list[str]:
            return re.split(r"(\s+)", text)

        orig_tokens = tokenize(original_text)
        tts_tokens = tokenize(tts_text)

        # SequenceMatcher lines the two texts up a word at a time. Each piece it
        # reports becomes a TextSegment, positioned in the original text.
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

            # A segment is all-or-nothing as soon as it holds a tag, so one tag in
            # the middle of otherwise identical text would hold the cursors still
            # for the whole piece. Giving the tag a segment of its own limits that
            # to the words the tag wraps:
            #
            #     original = tts = "I love to count <spell>1234</spell>."
            #
            #     "I love to count "       plain, cursors move word by word
            #     "<spell>1234</spell>."   all-or-nothing, lands as one
            #
            # Only equal pieces can be split this way. Both sides hold the same
            # text there, so a single position cuts both. Where the sides differ
            # there is no position that means the same thing on each.
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
    def _word_variants(word: str) -> tuple[str, ...]:
        """Return *word*, then *word* with any punctuation at its end removed.

        A TTS can add punctuation the text it was given never had -- reading a
        list item ``"my account"`` as a sentence and reporting ``"account."``.
        Matching tries the word as it arrived first, then the trimmed form.
        """
        trimmed = strip_trailing_punctuation(word)
        return (word,) if trimmed == word else (word, trimmed)

    @staticmethod
    def _literal_hop(
        candidates: list[tuple[str, int]],
        remaining_word: str,
        require_word_boundary: bool = False,
    ) -> "_Hop | None":
        """Compare *remaining_word* to each candidate, character for character.

        Two things can match. If a candidate starts with the word, the word fits
        here (``PLACED``). If the word starts with a candidate, the candidate ran
        out first, so the rest of the word belongs to the next segment
        (``CROSSES``).

        :meth:`_folded_hop` calls this too, on folded copies of the same strings.
        Folding never changes a string's length, so a position found in a folded
        copy is the same position in the original.

        Args:
            candidates: ``(text, offset)`` pairs to match *remaining_word*
                against, tried in order.
            remaining_word: The word (or its trailing-punctuation-stripped
                variant) to match.
            require_word_boundary: When True, the word must end where a word
                ends in the candidate -- either it used the candidate up, or the
                next character is not a letter or digit. This stops ``"account"``
                from matching the start of ``"Accountant"``. Only the folded pass
                asks for it, because folding away case makes that kind of
                accidental match much easier to hit.

        Returns:
            A ``PLACED`` or ``CROSSES`` hop, or ``None`` if nothing matched.
        """
        for word in TextSegmentMap._word_variants(remaining_word):
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
                        return _Hop(_HopKind.PLACED, segment_advance=offset + len(word))
                elif candidate and word.startswith(candidate):
                    return _Hop(_HopKind.CROSSES, word_consumed=len(candidate))
        return None

    @staticmethod
    def _leading_nonalnum_len(text: str, stop_at_markup: bool = False) -> int:
        """Count the characters at the start of *text* that are not letters or digits.

        For ``", I can"`` this returns 2 -- the comma and the space.

        With *stop_at_markup*, counting also stops at a ``'<'``, so the count
        never reaches inside a tag. Without it, ``"<break/>hello"`` would count
        past the ``'<'``, and a provider that reports the tag's name ``"break"``
        as a word would look like it had spoken it.
        """
        i = 0
        while i < len(text) and not text[i].isalnum():
            if stop_at_markup and text[i] == "<":
                break
            i += 1
        return i

    @staticmethod
    def _match_candidates(segment_remaining: str) -> list[tuple[str, int]]:
        """Return the three starting points a word may be matched against here.

        Providers disagree about what they include in a word, so the same text
        is offered from three places, each paired with how far in it starts:

        1. The text as it is, for a provider that reports ``" world"`` with its
           own leading space.
        2. Past any spaces.
        3. Past everything that is not a letter or digit, for a provider that
           does not repeat punctuation it already spoke -- when ``"I"`` arrives
           for ``"Yeah, I can"``, the ``", "`` is still waiting here.

        Each start is further in than the last, so the closest match is tried
        first. Identical starting points are dropped.
        """
        lead_ws = len(segment_remaining) - len(segment_remaining.lstrip())
        lead_nonalnum = TextSegmentMap._leading_nonalnum_len(segment_remaining, stop_at_markup=True)
        return [
            (segment_remaining[offset:], offset)
            for offset in dict.fromkeys((0, lead_ws, lead_nonalnum))
        ]

    @staticmethod
    def _folded_hop(candidates: list[tuple[str, int]], remaining_word: str) -> "_Hop | None":
        """Match *remaining_word* again, ignoring differences in how it is written.

        A provider may report a word in lower case, without accents, or with
        plain quotes: ``"SQL"`` as ``"sql"``, ``"café"`` as ``"cafe"``,
        ``"don’t"`` as ``"don't"``. Folding both sides makes those the same.

        Folding swaps characters one for one and never adds or removes any, so
        the strings keep their length and a position found here means the same
        position in the original text.

        Because folding hides case, a short word could now match inside a longer
        one -- ``"account"`` inside ``"Accountant"``. So a match here is only
        accepted if it ends where a word ends.
        """
        folded_candidates = [(fold_for_matching(c), offset) for c, offset in candidates]
        return TextSegmentMap._literal_hop(
            folded_candidates, fold_for_matching(remaining_word), require_word_boundary=True
        )

    @staticmethod
    def _markup_hop(segment_remaining: str, remaining_word: str) -> "_Hop | None":
        """Match *remaining_word* again, with tags removed from both sides.

        A provider may report a word wrapped in tags the text it was given did
        not have, or the other way round. Removing tags from both sides lets the
        words themselves be compared.

        The match is found in text that has no tags, so its position has to be
        translated back to a position in the real text, tags included, by
        :func:`~pipecat.utils.text.markup_utils.raw_offset_after_clean_chars`.

        Only ``PLACED`` comes out of this. A word that runs past the end of the
        segment is left to the two earlier passes.
        """
        stripped = segment_remaining.lstrip()
        lead_ws = len(segment_remaining) - len(stripped)
        haystack = strip_markup(stripped)

        for candidate in TextSegmentMap._word_variants(strip_markup(remaining_word)):
            if candidate and haystack.startswith(candidate):
                raw_len = raw_offset_after_clean_chars(stripped, len(candidate))
                return _Hop(_HopKind.PLACED, segment_advance=lead_ws + raw_len)
        return None

    @staticmethod
    def _classify_hop(segment_remaining: str, remaining_word: str) -> _Hop:
        """Decide what *remaining_word* does to the text left in this segment.

        Everything here is plain string comparison. No tag names are understood,
        and nothing is remembered between calls.

        Three ways of matching are tried, each more forgiving than the one
        before: :meth:`_literal_hop`, then :meth:`_folded_hop`, then
        :meth:`_markup_hop`. Any of them can report that the word fits here
        (``PLACED``) or that it runs past the end of the segment (``CROSSES``).

        If none of them match, the answer depends on what is left in the segment:

        - ``EXHAUSTED`` when nothing here can be spoken -- only a tag such as
          ``<break/>``, or trailing spaces and punctuation. The segment is
          finished so the word can try the next one. This is checked last, so a
          word that really does match something like a trailing emoji is found
          by the passes above first.
        - ``NO_MATCH`` otherwise. The word belongs somewhere else, so the cursor
          only steps past punctuation at the start of the segment, never past
          anything that was actually spoken.
        """
        candidates = TextSegmentMap._match_candidates(segment_remaining)

        hop = TextSegmentMap._literal_hop(candidates, remaining_word)
        if hop is None:
            hop = TextSegmentMap._folded_hop(candidates, remaining_word)
        if hop is None:
            hop = TextSegmentMap._markup_hop(segment_remaining, remaining_word)
        if hop is not None:
            return hop

        # Nothing left here that can be spoken: finish the segment so the word
        # can try the next one.
        if not has_alnum(segment_remaining):
            return _Hop(_HopKind.EXHAUSTED)

        # Foreign token: nudge past leading punctuation only, then stop. Unlike
        # the skip candidates this does not stop at markup -- it moves the raw
        # cursor rather than deciding a match, so there is no tag name it could
        # mistake for spoken content.
        return _Hop(
            _HopKind.NO_MATCH,
            segment_advance=TextSegmentMap._leading_nonalnum_len(segment_remaining),
        )

    def _advance_cursors_to(self, seg: TextSegment, new_pos: int) -> None:
        """Move every cursor to *new_pos* within *seg*, and finish *seg* if reached.

        This is where the "keep pace or wait" rule from the class docstring is
        applied, and the only place the cursors into the other two texts move.

        **Unchanged segment** -- both sides hold the same text, so the other two
        cursors keep pace. They move by a count, not a position: however many
        letters and digits this step used up in the spoken text is how many they
        move past. :func:`advance_by_alnums` does that, crossing tags for free
        (which is how a tag travels with the word beside it) and taking any
        punctuation stuck to the end of the word, so ``"you?"`` moves as one.

        **Rewritten segment** -- the other two cursors do not move at all until
        the segment is finished, then jump straight to its end.

        Either way a cursor stops before trailing spaces, which belong to
        whatever comes next.
        """
        if seg.is_transformed:
            # Whatever is left is only a closing tag or the like, which no word
            # event will ever name. Take it now so the segment can finish.
            # Unchanged segments are not given this: a trailing emoji there is
            # real output, and its own event is still coming.
            if not has_alnum(seg.tts[new_pos:]):
                new_pos = len(seg.tts)
        else:
            self._keep_derived_cursors_in_pace(seg, new_pos)

        self._seg_raw_pos = new_pos

        if new_pos >= len(seg.tts):
            if seg.is_transformed:
                self._commit_transformed_span(seg)
            self._finish_segment(seg)

    def _keep_derived_cursors_in_pace(self, seg: TextSegment, new_pos: int) -> None:
        """Move the cursors into the other two texts by what this step just spoke.

        The count of letters and digits consumed here is what they move by.
        """
        n_alnum = len(alnum_only(seg.tts[self._seg_raw_pos : new_pos]))
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

    def _commit_transformed_span(self, seg: TextSegment) -> None:
        """Jump the other two cursors to the end of *seg*, now that it is done."""
        self._user_facing_pos = seg.original_end
        # The original's count, not the TTS side's: llm_text holds "$42.50"
        # (4 alnums), never the spoken "forty two dollars".
        self._llm_pos = advance_by_alnums(self._llm_text, self._llm_pos, seg.original_alnum_count)

    def _finish_segment(self, seg: TextSegment) -> None:
        """Record *seg* as finished and move on to the next segment."""
        self._last_completed = seg
        self._seg_idx += 1
        self._seg_raw_pos = 0

    def _plan_hops(self, word: str) -> tuple[list[_Hop], str]:
        """Work out what *word* would do, without moving anything.

        This decides; :meth:`_consume_word` acts. Keeping them apart is what
        stops them disagreeing, since :meth:`_can_consume_word` asks the same
        question and must get the same answer.

        Most words are placed by the first segment tried, and the walk stops
        there. It goes on when a segment cannot finish the job -- the word runs
        past it, or it has nothing speakable left -- and whatever remains of the
        word is offered to the next segment.

        Returns:
            What each segment answered, in order, and whatever is left of the
            word once the segments run out. Anything left over is the word
            running past the end of this text.
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

            remaining_word = remaining_word[hop.word_consumed :]
            seg_idx += 1
            raw_pos = 0

        return hops, remaining_word

    def _consume_word(self, word: str) -> None:
        """Move the cursors according to what :meth:`_plan_hops` decided.

        Anything left unplaced ran past the end of this text and is kept in
        ``last_overflow``, for the caller to hand to the next frame.
        """
        hops, overflow = self._plan_hops(word)

        for hop in hops:
            seg = self._segments[self._seg_idx]

            if hop.kind is _HopKind.NO_MATCH:
                # The word belongs somewhere else entirely (a provider swapping a
                # symbol, say). Nudge the raw cursor past any leading punctuation
                # so the next word is not blocked by it, but leave the cursors
                # that mean something alone -- nothing was really spoken here.
                self._seg_raw_pos += hop.segment_advance
            elif hop.kind is _HopKind.PLACED:
                self._advance_cursors_to(seg, self._seg_raw_pos + hop.segment_advance)
            else:
                # CROSSES or EXHAUSTED: this segment is done either way, and the
                # next hop was classified against the one after it.
                self._advance_cursors_to(seg, len(seg.tts))

        if overflow:
            self._last_overflow = overflow

    def advance_word(self, word: str) -> None:
        """Take one spoken word and move every cursor to where it ends.

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
        """Count characters at the start of *word* that were already spoken.

        In ``"Yeah, I can"`` the comma travels with ``"Yeah"``. If the provider
        then reports the next word as ``", I"`` instead of ``"I"``, the comma
        would be recorded twice, so this returns 2 -- the comma and its space.

        Returns 0 when that punctuation is new text instead (``'"hello'``), and
        when the word is nothing but punctuation, which is a mark arriving on its
        own. Must be called before the cursors move, while ``llm_pos`` still sits
        at the end of the previous word.
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
        """Return True if *word* could be the next thing spoken here.

        :meth:`advance_word` without the moving, so a caller can check first. A
        False answer means the provider skipped ahead, and the word should go to
        the next frame instead.

        A word with no letters or digits gets a second chance from
        :meth:`_symbol_belongs_here`, since there is nothing in it to match on.
        """
        if not word:
            return True
        if self._can_consume_word(word):
            return True
        if not has_alnum(word):
            return self._symbol_belongs_here(word)
        return False

    def _can_consume_word(self, word: str) -> bool:
        """Ask whether this word would be placed, without placing it.

        True if some segment would take the word, or if the word simply runs off
        the end. False if there are no segments left, or a segment turns it down.
        """
        if self._seg_idx >= len(self._segments):
            return False

        hops, _ = self._plan_hops(word)
        return not hops or hops[-1].kind is not _HopKind.NO_MATCH

    def _symbol_belongs_here(self, word: str) -> bool:
        """Decide whether a word made only of punctuation or symbols belongs here.

        There is nothing in such a word to match on, so it gets two chances:

        1. Look for the word itself in the text still to be spoken. The search
           starts a little before the cursor, because punctuation is often taken
           along with the word before it.

        2. Accept it as a stand-in. Some providers report a different symbol
           than the one they were given -- ElevenLabs reports ``"->"`` as
           ``"-"`` -- so the first check can never succeed. If words remain to be
           spoken and the next thing in the text is itself a symbol, treat the
           word as that symbol.
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
        """How far into the user-facing text the spoken words have reached."""
        return self._user_facing_pos

    @property
    def llm_pos(self) -> int:
        """How far into the LLM's text the spoken words have reached."""
        return self._llm_pos

    @property
    def raw_pos(self) -> int:
        """How far into ``tts_text`` the provider has spoken, counted from its start."""
        pos = sum(len(s.tts) for s in self._segments[: self._seg_idx])
        if self._seg_idx < len(self._segments):
            pos += self._seg_raw_pos
        return pos

    @property
    def last_overflow(self) -> str | None:
        """The end of the last word passed to :meth:`advance_word`, if it did not fit.

        ``None`` most of the time. It is set only when that word ran past the
        end of ``tts_text``, with no segment left to take the rest, which means
        the leftover belongs to the next frame. It is always the tail of the
        word that was passed in, so the part that did fit is
        ``word[: len(word) - len(last_overflow)]``.
        """
        return self._last_overflow

    @property
    def last_leading_duplicate(self) -> int:
        """How much of the last word's start was punctuation already spoken.

        The opposite end of the word from :attr:`last_overflow`: that one is
        about a tail running past this text, this one about a head repeating
        punctuation the previous word already took. Cut both off to get the part
        of the word that belongs to this frame::

            word[last_leading_duplicate : len(word) - len(last_overflow or "")]
        """
        return self._last_leading_duplicate

    @property
    def is_complete(self) -> bool:
        """True once every letter and digit in the text has been spoken.

        This is not the same as the cursor reaching the end. If all that is left
        is punctuation or tags, the text counts as finished even though those
        characters have not been walked over, because no word event is coming
        for them.

        There is one exception. Punctuation separated from its word by a space,
        as French writes ``"Comment ça va ?"``, does arrive as its own word
        event, so the text stays unfinished until it does (see
        :meth:`_pending_separated_punctuation`). Punctuation stuck to the word
        itself, as in ``"you?"``, was already taken with the word.
        """
        if self._seg_idx >= len(self._segments):
            return True
        seg = self._segments[self._seg_idx]
        if has_alnum(seg.tts[self._seg_raw_pos :]):
            return False
        if self._pending_separated_punctuation(seg.tts[self._seg_raw_pos :]):
            return False
        return all(not has_alnum(s.tts) for s in self._segments[self._seg_idx + 1 :])

    @staticmethod
    def _pending_separated_punctuation(remaining: str) -> bool:
        """True when all that is left is punctuation set off from its word by a space.

        Some languages write a space before a mark, as in ``"va ?"`` or
        ``"Bonjour !"``. A TTS reports that mark as a word of its own, so the
        segment has to stay open until it arrives.

        Only real punctuation counts. A trailing emoji or arrow (``"day! 😊"``,
        ``"→"``) never arrives as its own word, so it must not keep the segment
        open, and tags are removed first for the same reason. Only called once
        no letters or digits are left.
        """
        stripped_markup = strip_complete_markup(remaining)
        if not stripped_markup[:1].isspace():
            return False
        content = stripped_markup.strip()
        return bool(content) and unicodedata.category(content[0]).startswith("P")

    @property
    def in_transformed_segment(self) -> bool:
        """True when the cursor is partway through a rewritten segment."""
        if self._seg_idx >= len(self._segments):
            return False

        seg = self._segments[self._seg_idx]
        return seg.is_transformed and self._seg_raw_pos > 0

    @property
    def last_completed_segment(self) -> TextSegment | None:
        """The segment finished by the last :meth:`advance_word` call, if any."""
        return self._last_completed

    def reset(self) -> None:
        """Put every cursor back to the start of the text."""
        self._reset_state()
