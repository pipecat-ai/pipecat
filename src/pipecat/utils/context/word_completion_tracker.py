#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Per-frame bookkeeping for the words a TTS provider reports speaking."""

from loguru import logger

from pipecat.utils.context.text_segment_map import TextSegmentMap
from pipecat.utils.text.markup_utils import strip_complete_markup


class WordCompletionTracker:
    """Follows one AggregatedTextFrame from dispatch until it is fully spoken.

    A TTS provider reports the words it speaks one event at a time. This class
    consumes those events for a single frame and answers, for each one:

    - What text should the emitted ``TTSTextFrame`` carry? -- :meth:`get_word_for_frame`
    - Which of the LLM's own text does that stand for? -- :meth:`get_llm_consumed`
    - How much of the frame has been spoken? -- :meth:`get_accumulated_user_facing_text`
      and :meth:`get_remaining_user_facing_text`
    - Is the frame finished? -- the return of :meth:`add_word_and_check_complete`
    - Did the word run past this frame? -- :meth:`get_overflow_word`

    Three texts describe the same frame, and each answer above is phrased in one
    of them:

    ===================== ============================= =======================
    Text                  Example                       Answers about
    ===================== ============================= =======================
    ``tts_text``          ``<spell>4111 1111</spell>``  what was spoken
    ``user_facing_text``  ``4111 1111``                 what a UI displays
    ``llm_text``          ``<card>4111 1111</card>``    what the context stores
    ===================== ============================= =======================

    Keeping a position in all three is the job of
    :class:`~pipecat.utils.context.text_segment_map.TextSegmentMap`, which this
    class owns one of and defers to for every question of *where*. What the
    tracker adds is the handful of decisions a position alone cannot express:

    - **Providers drop events.** When a word does not match what is left to
      speak, waiting for it would stall this frame and everything queued behind
      it. The frame is force-completed instead: the unspoken remainder is emitted
      so the context still gets it, and the stray word is handed back for the
      next frame to try.
    - **Some text is never spoken.** A closing ``</card>``, or a tag sitting
      between the last word and its punctuation, never arrives as its own event.
      Whatever is left once everything speakable is done belongs to this frame,
      so the word that finishes it claims the rest.
    - **A word can belong to two frames.** A provider may merge across the
      boundary (``"1111And"``). The part that fits stays here; the rest is
      exposed as overflow for the caller to feed to the next frame.

    Example::

        tracker = WordCompletionTracker("Hello, world!")
        tracker.add_word_and_check_complete("Hello")   # False
        tracker.add_word_and_check_complete("world")   # True -- nothing left to speak
    """

    def __init__(
        self,
        tts_text: str,
        llm_text: str | None = None,
        user_facing_text: str | None = None,
    ):
        """Initialize the tracker with the frame's three texts.

        Only ``tts_text`` is required; the other two default to it, which is
        exactly right for a frame nothing rewrote.

        Args:
            tts_text: What was sent to the TTS, and so what the incoming words
                are matched against. May carry synthesis tags (``<spell>...``).
            llm_text: What the LLM wrote, with any delimiters an aggregator split
                off (``<card>4111 1111</card>``). Supply it to have each word
                attributed back to it via :meth:`get_llm_consumed`, which is what
                keeps those delimiters in the conversation context.
            user_facing_text: What a client displays -- no tags, no rewrites.
                Defaults to ``tts_text`` with markup stripped.
        """
        # --- The three texts ---
        self._tts_text = tts_text
        # Stripping markup from the fallback keeps synthesis tags out of what a UI
        # shows, and gives the map a plain run to split around any tag rather than
        # one atomic string.
        self._user_facing_text: str = (
            user_facing_text if user_facing_text is not None else strip_complete_markup(tts_text)
        )
        self._llm_text = llm_text

        # --- Cursors into two of them ---
        # The map owns the authoritative positions; these mirror it, and only
        # diverge where the tracker deliberately moves further (see
        # _record_llm_span and _force_complete). The position in tts_text is not
        # mirrored -- the map's raw_pos is read directly.
        self._user_facing_pos = 0
        self._llm_pos = 0

        # --- Answers about the most recent word ---
        # Rewritten by every add_word_and_check_complete call and read back
        # through the get_* accessors.
        self._frame_word: str | None = None  # this frame's share of the word
        self._overflow_word: str | None = None  # the next frame's share
        self._llm_consumed: str | None = None  # LLM text the word stands for

        # Set by _force_complete. The map is never advanced there, so its own
        # is_complete would keep saying False; this flag is the answer instead.
        self._force_completed = False

        self._segment_map = TextSegmentMap(tts_text, self._user_facing_text, llm_text)

    def add_word_and_check_complete(self, word: str) -> bool:
        """Record one word the TTS provider reported speaking.

        Three things can happen, in this order:

        1. The frame is already finished -- the word is ignored.
        2. The word does not match what is left to speak, so the provider must
           have dropped an event: the frame is force-completed (see
           :meth:`_force_complete`) and this word is handed back as overflow.
        3. Otherwise the word advances the frame. Afterwards the ``get_*``
           accessors describe it: this frame's share of the word, the LLM text it
           stands for, and how much of the frame is now spoken.

        Args:
            word: One token from the provider's word-timestamp stream. It may be
                a plain word, a word carrying its own spacing or punctuation, or
                a fragment of a still-open SSML tag -- matching is textual, so
                none of those need special handling from the caller. Services
                that report spaces and punctuation as separate tokens (e.g.
                Inworld) must merge them into the preceding word first, via
                ``merge_punct_tokens``.

        Returns:
            True once nothing is left for this frame to speak.
        """
        self._frame_word = None
        self._overflow_word = None
        self._llm_consumed = None

        # Every raw character consumed, not is_complete: a frame ending in an
        # emoji contributes no alphanumeric content, so it reads as complete
        # before that emoji's own event arrives, and that event is still wanted.
        if self._force_completed or self._segment_map.raw_pos >= len(self._tts_text):
            logger.warning(f"{self}, trying to add a word in an already complete frame")
            return True

        if not self.word_belongs_here(word):
            return self._force_complete(word)

        llm_pos_before = self._llm_pos
        self._segment_map.advance_word(word)

        # Neither end of the token is necessarily this frame's: the head can
        # repeat punctuation the previous word already carried, and the tail can
        # run into the next frame. The map measures both; keep what is between.
        # Without an llm_text there is no recorded span that could already have
        # carried the mark, so it is new text on this frame.
        head = self._segment_map.last_leading_duplicate if self._llm_text is not None else 0
        overflow = self._segment_map.last_overflow
        tail = len(word) - len(overflow) if overflow else len(word)
        self._frame_word = word[head:tail]
        self._overflow_word = overflow

        self._user_facing_pos = self._segment_map.user_facing_pos
        self._llm_pos = self._segment_map.llm_pos

        if self._llm_text is not None:
            self._record_llm_span(word, llm_pos_before)

        complete = self.is_complete
        if complete:
            # Everything speakable has been spoken, so anything still left is
            # text no word will ever arrive for -- a closing tag, or one sitting
            # between the last word and its punctuation. It belongs to this
            # frame, so take it rather than leave it out of the turn.
            self._user_facing_pos = len(self._user_facing_text)
        return complete

    def _force_complete(self, word: str) -> bool:
        """End this frame early because *word* does not belong to it.

        The provider dropped one or more events, so the rest of this frame will
        never be reported. Rather than stall, emit the unspoken remainder as this
        frame's word -- the conversation context still receives the full text --
        and hand *word* back as overflow for the next frame to try.

        The segment map is deliberately left where it is, since nothing here was
        actually spoken; :attr:`_force_completed` answers for it from now on.

        Returns:
            Always True -- the frame is finished.
        """
        self._frame_word = self._tts_text[self._segment_map.raw_pos :]
        self._user_facing_pos = len(self._user_facing_text)
        if self._llm_text is not None:
            # The whole remainder is this frame's by definition, tags included.
            self._llm_consumed = self._llm_text[self._llm_pos :]
            self._llm_pos = len(self._llm_text)
        self._force_completed = True
        self._overflow_word = word
        return True

    def _record_llm_span(self, word: str, llm_pos_before: int) -> None:
        """Record which part of ``llm_text`` the word just added stands for.

        Usually that is simply the span the map's cursor moved over. Two cases
        reach further, and both leave ``_llm_pos`` ahead of the map's:

        - **The word finished the frame**: take everything to the end of
          ``llm_text``. The map stops at the last spoken character, so a closing
          tag -- which never arrives as its own event -- is still outstanding and
          belongs to this word.
        - **The cursor did not move**, because the map placed the word without
          spending any budget (an emoji or symbol): take the word's own length
          from ``llm_text``, skipping spaces the previous word owns.

        A word inside a transformed segment records nothing, and is checked
        before that second case: the cursor is held there on purpose, so "did not
        move" would be misread as "spent nothing" and would walk the cursor
        through text the transform covers. Only the word completing the segment
        carries its original span.
        """
        assert self._llm_text is not None

        if self.is_complete:
            self._llm_consumed = self._llm_text[llm_pos_before:]
            self._llm_pos = len(self._llm_text)
        elif self._segment_map.in_transformed_segment:
            self._llm_consumed = None
        elif self._llm_pos == llm_pos_before and self._segment_map.last_completed_segment is None:
            start = self._llm_pos
            while start < len(self._llm_text) and self._llm_text[start].isspace():
                start += 1
            end = start + len(word)
            self._llm_consumed = self._llm_text[start:end]
            self._llm_pos = end
        else:
            self._llm_consumed = self._llm_text[llm_pos_before : self._llm_pos]

    def word_belongs_here(self, word: str) -> bool:
        """Return True if *word* plausibly continues what this frame has left to say.

        A False answer means the provider dropped an event. Callers ask this
        before adding a word so they can offer it to the next frame instead;
        adding it anyway force-completes this one.
        """
        return self._segment_map.word_belongs_current_segment(word)

    def suppress_in_context(self) -> bool:
        """True when the last word was one step inside a rewritten span.

        ``"$42.50"`` is spoken as five words, none of which the transcript should
        contain. Callers keep every such word out of the conversation context and
        let the word that finishes the span carry ``"$42.50"`` for all of them.
        """
        return self._segment_map.in_transformed_segment

    def get_word_for_frame(self) -> str | None:
        """Return this frame's share of the last word -- the text to emit for it.

        Usually the whole word. A word straddling the boundary gives up its tail
        (``"1111"`` out of ``"1111And"``), and a word that repeats the previous
        word's punctuation gives up that mark. After a force-complete this is the
        frame's unspoken remainder instead, so nothing is missing from the turn.
        """
        return self._frame_word.strip() if self._frame_word else self._frame_word

    def get_overflow_word(self) -> str | None:
        """Return the part of the last word that belongs to the *next* frame.

        Feed it to that frame's tracker as if the provider had sent it there.
        Casing and punctuation are untouched so it still reads as a real word.
        None when the word fit entirely within this frame.
        """
        return self._overflow_word.strip() if self._overflow_word else self._overflow_word

    def get_llm_consumed(self) -> str | None:
        """Return the LLM's own text that the last word stands for.

        This is what the conversation context records, so it keeps the tags and
        spellings the LLM wrote (``"<card>4111"``) rather than what the provider
        reported speaking (``"4111"``).

        None when there is nothing to attribute: no ``llm_text`` was given, the
        word is mid-rewrite, or ``llm_text`` is already exhausted (a trailing
        emoji it never carried).
        """
        if not self._llm_consumed:
            return None
        return self._llm_consumed.strip() or None

    def get_accumulated_user_facing_text(self) -> str:
        """Return the part of the frame spoken so far, as the user sees it.

        With :meth:`get_remaining_user_facing_text` this splits the frame's text
        in two, which is what lets a client highlight speech as it happens.
        """
        return self._user_facing_text[: self._user_facing_pos]

    def get_remaining_user_facing_text(self, strip: bool = True) -> str:
        """Return the part of the frame not yet spoken, as the user sees it.

        Args:
            strip: Whether to trim surrounding whitespace. Pass False to keep the
                leading space, so accumulated + remaining reproduces the frame's
                text exactly -- callers that index into that text rely on it.
        """
        remaining = self._user_facing_text[self._user_facing_pos :]
        return remaining.strip() if strip else remaining

    def get_accumulated_tts_text(self) -> str:
        """Return everything spoken so far, as it was sent to the TTS.

        The whole frame up to the cursor, where :meth:`get_word_for_frame`
        describes only the most recent word.
        """
        return self._tts_text[: self._segment_map.raw_pos]

    def get_accumulated_llm_text(self) -> str | None:
        """Return everything spoken so far, as the LLM wrote it.

        The whole frame up to the cursor, where :meth:`get_llm_consumed`
        describes only the most recent word. None without an ``llm_text``.
        """
        if self._llm_text is None:
            return None
        return self._llm_text[: self._llm_pos]

    def get_remaining_tts_text(self, strip: bool = True) -> str:
        """Return what this frame still has left to speak.

        Callers ending a frame early emit this, so text the provider never
        reported still reaches the conversation context.

        Args:
            strip: Whether to trim surrounding whitespace. Pass False to keep the
                leading space, so accumulated + remaining reproduces ``tts_text``
                exactly.
        """
        remaining = self._tts_text[self._segment_map.raw_pos :]
        return remaining.strip() if strip else remaining

    def get_remaining_llm_text(self) -> str | None:
        """Return what this frame still has left to speak, as the LLM wrote it.

        The companion to :meth:`get_remaining_tts_text` when ending a frame
        early: that supplies the text to emit, this the text to record. None
        without an ``llm_text``, or when nothing is left.
        """
        if self._llm_text is None:
            return None
        remaining = self._llm_text[self._llm_pos :].strip()
        return remaining if remaining else None

    @property
    def is_complete(self) -> bool:
        """True when this frame has nothing left to speak.

        Alphanumeric content is what counts, so a frame whose remainder is only
        punctuation or a closing tag is already finished -- no word will ever
        arrive for those.
        """
        return self._force_completed or self._segment_map.is_complete

    def reset(self):
        """Rewind to the start of the frame, keeping the three texts."""
        self._user_facing_pos = 0
        self._llm_pos = 0
        self._overflow_word = None
        self._llm_consumed = None
        self._frame_word = None
        self._force_completed = False
        self._segment_map.reset()
