#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ordered sequencer for AggregatedTextFrame slots through TTS processing."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    AggregationType,
    Frame,
    TTSTextFrame,
)
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker
from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


@dataclass
class _ParallelAggregation:
    """One completed sentence, in each of the sequencer's three parallel channels.

    Parameters:
        tts_text: The sentence as sent to the TTS service (post-filter/transform).
        llm_text: The sentence in the original LLM text (with any pattern delimiters).
        user_facing_text: The sentence as shown to the user (no TTS tags/transforms).
    """

    tts_text: str
    llm_text: str
    user_facing_text: str


class _ParallelSentenceAggregator:
    """Internal: groups streamed tokens back into sentences for the sequencer.

    Used by :class:`AggregatedFrameSequencer` when a TTS service streams tokens
    individually (``TextAggregationMode.TOKEN``) but still needs whole-sentence
    units for word-timestamp tracking and RTVI progress. Each token contributes
    one part to each of the three channels (tts / llm / user-facing); sentence
    completion is driven by the **TTS text** and completes all three together.

    Boundary timing: a sentence-ending boundary is only *confirmed* by lookahead —
    the first non-whitespace character of the *next* sentence. So the underlying
    :class:`SimpleTextAggregator` only yields "Hi there!" once the following
    non-whitespace character has arrived.

    Splitting granularity: token streams are not guaranteed to be one word or one
    punctuation mark per token — an upstream can deliver a coarse chunk that carries
    the tail of one sentence *and* the head of the next (e.g. "Hey" then
    " there! I'm ..."). Two cases:

    - When every token accumulated for the pending sentence has identical text in
      all three channels (the common case — no TTS transform/tag rewriting is
      active), a boundary is sliced *inside* the triggering token at its confirmed
      offset, so " there!" stays with "Hey" and only " I'm ..." carries over.
    - When a transform has made the channels differ in length, the channels can no
      longer be split at a shared character offset, so the boundary is cut at the
      token boundary instead: the text accumulated *before* the triggering token is
      emitted and the whole triggering token begins the next sentence's buffer.
    """

    def __init__(self):
        """Initialize the aggregator with empty channels."""
        # A plain SENTENCE-mode aggregator drives boundary detection on the TTS
        # text. The TTS text is already post-transform, so tag/pattern-aware
        # boundary rules are not needed here.
        self._aggregator = SimpleTextAggregator(aggregation_type=AggregationType.SENTENCE)
        self._reset()

    def _reset(self):
        # Tokens accumulated since the last emitted sentence, per channel.
        self._tts = ""
        self._llm = ""
        self._user = ""
        # Whether every token accumulated for the pending sentence has identical
        # text across the three channels. While True the tts buffer mirrors the
        # inner aggregator's buffer, so a boundary can be sliced inside a token.
        self._aligned = True

    async def aggregate(
        self, tts_text: str, llm_text: str, user_facing_text: str
    ) -> AsyncIterator[_ParallelAggregation]:
        """Feed one token (all three channels) and yield any completed sentences.

        Args:
            tts_text: The token as sent to the TTS service.
            llm_text: The token in the original LLM text.
            user_facing_text: The token as shown to the user.

        Yields:
            A :class:`_ParallelAggregation` for each sentence this token confirms.
            Usually zero or one, but a coarse chunk carrying several sentence
            endings can confirm more than one at once.
        """
        token_identical = tts_text == llm_text == user_facing_text

        # The inner SENTENCE aggregator is the source of truth for *when* and *how
        # many* boundaries are confirmed (it applies the lookahead rule char by
        # char on the TTS text).
        boundary_count = 0
        async for _ in self._aggregator.aggregate(tts_text):
            boundary_count += 1

        if boundary_count and self._aligned and token_identical:
            # Channels are identical for the whole pending sentence, so the tts
            # buffer mirrors the inner aggregator's buffer and we can slice inside
            # this token at each confirmed boundary, keeping the channels aligned.
            combined = self._tts + tts_text
            idx = 0
            for _ in range(boundary_count):
                boundary = match_endofsentence(combined[idx:])
                if boundary <= 0:
                    break
                sentence = combined[idx : idx + boundary]
                yield _ParallelAggregation(sentence, sentence, sentence)
                idx += boundary
            # The remainder came from identical channels, so the next pending
            # sentence starts aligned and its buffer still mirrors the inner one.
            self._tts = self._llm = self._user = combined[idx:]
            self._aligned = True
            return

        # A transform has diverged the channels (or nothing is buffered before this
        # token): we can only cut at the token boundary. Emit whatever was buffered
        # before this token and let this token begin the next sentence's buffer.
        if boundary_count and (self._tts or self._llm or self._user):
            yield _ParallelAggregation(self._tts, self._llm, self._user)
            self._tts = self._llm = self._user = ""

        # Plain concatenation: LLM tokens already carry their own spacing.
        self._tts += tts_text
        self._llm += llm_text
        self._user += user_facing_text
        # Re-derive alignment from ground truth rather than assuming a token-boundary
        # cut restores it.
        self._aligned = self._tts == self._llm == self._user and self._tts == self._aggregator._text

    async def flush(self) -> _ParallelAggregation | None:
        """Emit any trailing partial sentence at end of turn.

        Returns:
            A :class:`_ParallelAggregation` for the accumulated-but-unemitted text,
            or ``None`` when nothing substantive is buffered.
        """
        await self._aggregator.flush()
        if self._user.strip():
            result = _ParallelAggregation(self._tts, self._llm, self._user)
            self._reset()
            return result
        return None

    async def handle_interruption(self):
        """Discard all buffered text (called on interruption/reset)."""
        await self._aggregator.handle_interruption()
        self._reset()


@dataclass
class _AggregatedFrameSlot:
    """Ordered slot tracking one AggregatedTextFrame through TTS processing.

    Every frame that passes through _push_tts_frames — whether spoken or skipped —
    occupies a slot in the sequencer. Skipped frames wait at their position and are
    emitted downstream only after all preceding spoken slots are complete, preserving
    correct context ordering.
    """

    frame: AggregatedTextFrame
    context_id: str
    spoken: bool
    tracker: WordCompletionTracker | None = None
    transport_destination: str | None = None
    complete: bool = False
    includes_inter_frame_spaces: bool = False


@dataclass
class _BufferedWord:
    """A word-timestamp event held until the slot it belongs to is promoted.

    In streaming mode a word can arrive before the sentence it belongs to has been
    promoted into a real slot. The event is parked here and replayed once a matching
    slot appears (see :meth:`AggregatedFrameSequencer._drain_buffered_words`).
    """

    word: str
    pts: int
    context_id: str | None
    includes_inter_frame_spaces: bool


@dataclass
class _StreamingContext:
    """Per-context streaming (TOKEN-mode) state.

    Each concurrently-live audio context keeps its own token→sentence aggregator and
    the slot metadata a promoted sentence needs, so two contexts in flight at once
    never share one pending-sentence buffer.

    Parameters:
        aggregator: Groups this context's streamed tokens back into sentences.
        append_to_context: Whether word frames built for a promoted sentence should
            carry ``append_to_context=True``. Turn-constant for the context.
    """

    aggregator: "_ParallelSentenceAggregator"
    append_to_context: bool


class AggregatedFrameSequencer:
    """Sequences AggregatedTextFrame slots to preserve TTS context ordering.

    Manages an ordered queue of spoken and skipped TTS slots. Spoken slots are tracked
    via a :class:`WordCompletionTracker`; skipped slots (e.g. code blocks excluded from
    TTS synthesis) wait in-place until all preceding spoken slots are complete, then are
    flushed downstream with ``append_to_context=True``.

    Most methods are synchronous and return lists of frames the caller should push
    downstream, making the sequencer easily testable. The exceptions are
    :meth:`register_spoken`, :meth:`register_skipped`, and :meth:`finalize`, which
    are async because — when the sequencer is built with ``streaming=True`` — they
    drive an async :class:`_ParallelSentenceAggregator` to group streamed tokens into
    sentences.

    Contexts can be in flight concurrently (e.g. two back-to-back ``TTSSpeakFrame``
    utterances on a websocket TTS service whose ``run_tts`` returns before synthesis
    finishes). State is organised in three tiers so concurrent contexts never bleed
    into each other:

    - ``_slots``: the single global ordered timeline across all contexts. Ordering is
      the whole point of the sequencer, so this stays one list.
    - ``_context_append_to_context``: a per-context flag whose presence also marks the
      context as *live* — an entry existing is the "process this context's words"
      signal in :meth:`process_word`'s stale gate. Created at slot registration,
      removed when the context is fully done (:meth:`force_complete`).
    - ``_streaming_contexts``: transient per-context pending-sentence state
      (:class:`_StreamingContext`), present only while a sentence is being accumulated
      from tokens. Released at end of text input (:meth:`finalize`).

    Example::

        sequencer = AggregatedFrameSequencer()
        await sequencer.register_spoken(frame, ctx_id, tts_text, append_to_context=True)
        for f in sequencer.process_word("hello", pts=1000, context_id=ctx_id):
            await self.push_frame(f)
    """

    def __init__(self, name: str = "AggregatedFrameSequencer", streaming: bool = False):
        """Initialize the sequencer.

        Args:
            name: Label used in log messages (typically the owning TTS service name).
            streaming: True when tokens are dispatched to the TTS individually
                (``TextAggregationMode.TOKEN``). Each :meth:`register_spoken` call
                then represents one token rather than a complete unit, so tokens
                are fed to a :class:`_ParallelSentenceAggregator` and only turned into a
                real slot once a sentence boundary is detected (or forced via
                :meth:`register_skipped`/:meth:`finalize`). Fixed for the life of
                the sequencer — a TTS service's aggregation mode never changes at
                runtime.

                Requires the owning TTS service to reuse one context ID for the whole
                turn (``reuse_context_id_within_turn=True``, the default): a promoted
                sentence built from several tokens is registered under a single
                context ID, and word-timestamp events for all of its tokens must
                arrive tagged with that same ID. A per-token context ID would leave
                every token but the last unmatched, and its words dropped as stale.
        """
        self._name = name
        self._streaming = streaming
        self._slots: list[_AggregatedFrameSlot] = []
        self._context_append_to_context: dict[str, bool] = {}
        # Transient per-context pending-sentence state, keyed by context ID. An entry
        # exists only while a sentence is being accumulated from tokens for that
        # context (streaming mode only). Aggregators are created lazily per context.
        self._streaming_contexts: dict[str, _StreamingContext] = {}
        self._buffered_words: list[_BufferedWord] = []

    async def register_spoken(
        self,
        frame: AggregatedTextFrame,
        context_id: str,
        tts_text: str,
        append_to_context: bool,
        build_tracker: bool = True,
        includes_inter_frame_spaces: bool = False,
    ) -> list[Frame]:
        """Register a spoken AggregatedTextFrame slot.

        Called from _push_tts_frames for every frame sent to the TTS service (one
        call per token when ``streaming=True``, one call per complete unit
        otherwise). Builds the :class:`WordCompletionTracker` internally when
        ``build_tracker`` is set — callers never construct one themselves. A
        registered slot is marked complete either via :meth:`process_word`
        (word-timestamp services) or :meth:`complete_spoken_slot`
        (push_text_frames=True services).

        When the sequencer is non-streaming, or streaming without a tracker
        (push_text_frames=True providers), this registers a slot immediately. When
        streaming with a tracker, the call instead feeds this token to the
        :class:`_ParallelSentenceAggregator` and only registers a real slot once a
        sentence boundary is confirmed there.

        Args:
            frame: The AggregatedTextFrame being spoken (one token when streaming).
            context_id: The TTS context ID assigned to this frame.
            tts_text: The text actually sent to the TTS for this call (may differ
                from ``frame.text`` after filters/transforms).
            append_to_context: Whether word frames built for this context should carry
                append_to_context=True.
            build_tracker: Whether to track word completion at all. False for
                push_text_frames=True services, which complete via
                complete_spoken_slot instead of word-timestamp matching.
            includes_inter_frame_spaces: When True, every TTSTextFrame emitted for this
                slot carries ``includes_inter_frame_spaces=True`` so downstream consumers
                do not inject extra spaces between consecutive frames. Not used on the
                streaming path — there, CJK spacing is driven solely by
                :meth:`process_word`'s per-call flag.

        Returns:
            Frames unblocked by this call (buffered words replayed once a pending
            sentence promotes). Always empty for the non-streaming and
            no-tracker cases.
        """
        if not self._streaming or not build_tracker:
            self._append_spoken_slot(
                frame,
                context_id,
                WordCompletionTracker(
                    tts_text,
                    llm_text=frame.raw_text or frame.text,
                    user_facing_text=frame.text,
                )
                if build_tracker
                else None,
                append_to_context,
                includes_inter_frame_spaces,
            )
            return []

        sc = self._streaming_contexts.setdefault(
            context_id, _StreamingContext(_ParallelSentenceAggregator(), append_to_context)
        )
        frames: list[Frame] = []
        async for agg in sc.aggregator.aggregate(
            tts_text, frame.raw_text or frame.text, frame.text
        ):
            frames.extend(self._promote(agg, context_id, sc.append_to_context))
        return frames

    async def register_skipped(
        self,
        frame: AggregatedTextFrame,
        context_id: str,
        transport_destination: str | None,
    ) -> list[Frame]:
        """Register a skipped AggregatedTextFrame and attempt an immediate flush.

        Any sentence still pending for this context is finalized first, so a real
        spoken slot exists immediately before the skipped slot in the queue —
        :meth:`flush`'s "stop at first incomplete spoken slot" logic then blocks
        this skipped frame correctly until that sentence is actually spoken.

        The frame is appended as a skipped slot. If no incomplete spoken slot precedes
        it, the frame is returned right away; otherwise it waits until a later
        :meth:`flush` unblocks it.

        Args:
            frame: The skipped AggregatedTextFrame (e.g. a code block).
            context_id: The context ID assigned in _push_tts_frames.
            transport_destination: Transport routing value to attach at flush time.

        Returns:
            Frames to push downstream: any sentence promoted by the initial
            :meth:`finalize` (streaming mode), followed by this skipped frame once it
            is unblocked. The skipped frame itself is absent while a preceding spoken
            slot is still incomplete — the promoted-sentence frame can still be
            returned in that case, so the list is not necessarily empty when blocked.
        """
        frames = await self.finalize(context_id)
        frame.context_id = context_id
        self._slots.append(
            _AggregatedFrameSlot(
                frame=frame,
                context_id=context_id,
                spoken=False,
                transport_destination=transport_destination,
            )
        )
        frames.extend(self.flush())
        return frames

    async def finalize(self, context_id: str | None) -> list[Frame]:
        """Force-promote a context's still-pending sentence into a real slot.

        Called at end of text input for a context (no more tokens are coming), to
        handle a response that ends with no terminal punctuation. The context's
        pending-sentence state is released here; its ``_context_append_to_context``
        entry is deliberately kept, since word-timestamp events for the just-promoted
        slot still arrive later (during audio playback) and must be recognised as
        live. A no-op when nothing is pending for the context (or the sequencer is
        not streaming).

        Args:
            context_id: The context whose pending sentence should be finalized.

        Returns:
            Frames unblocked by finalizing (e.g. buffered words that can now
            be replayed against the newly-registered slot).
        """
        if not self._streaming or context_id is None:
            return []
        sc = self._streaming_contexts.pop(context_id, None)
        if sc is None:
            return []
        agg = await sc.aggregator.flush()
        return self._promote(agg, context_id, sc.append_to_context) if agg else []

    def process_word(
        self,
        word: str,
        pts: int,
        context_id: str | None,
        includes_inter_frame_spaces: bool = False,
    ) -> list[Frame]:
        """Process one word-timestamp event and return frames to push downstream.

        Locates the active slot (the first incomplete spoken slot with a tracker for
        this ``context_id``), advances it by the incoming word, and builds a
        :class:`TTSTextFrame`. Handles:

        - Words from a context that was never registered, was wiped by :meth:`clear`
          on interruption, or has already completed (:meth:`force_complete` removes
          its entry): dropped as stale (returns an empty list).
        - Normal words that fit entirely within the active slot.
        - Overflow words straddling two slot boundaries.
        - Force-complete when the TTS drops an event (word belongs to the next slot).
        - Passthrough for words not recognised by any slot (buffered instead, when
          streaming, since the slot they belong to may simply not be promoted yet).
        - Flushes any skipped slots unblocked by slot completion.

        Args:
            word: A word token from the TTS service word-timestamp stream.
            pts: Presentation timestamp (nanoseconds) to assign to the frame.
            context_id: TTS context ID from the word-timestamp event.
            includes_inter_frame_spaces: Stamped onto the emitted TTSTextFrame so
                downstream consumers know not to inject extra spaces between frames.

        Returns:
            Ordered list of frames (TTSTextFrame and/or AggregatedTextFrame) to push.
        """
        # Drop words from contexts we never registered, that were wiped by clear() on
        # interruption, or that have already fully completed (force_complete removes the
        # entry). Such a word is stale (e.g. delayed word-timestamps the TTS server
        # delivers seconds after the context was cancelled); emitting it would interleave
        # it into the current turn's transcript. A None context_id is left untouched:
        # services without audio contexts legitimately use the passthrough path below. A
        # word for a context still streaming a pending sentence (no slot promoted yet) is
        # not stale — it's handled by the buffering below.
        is_pending_streaming_ctx = context_id in self._streaming_contexts
        if (
            context_id is not None
            and context_id not in self._context_append_to_context
            and not is_pending_streaming_ctx
        ):
            logger.debug(
                f"{self._name} Dropping stale word '{word}' from unknown/cleared "
                f"context {context_id}"
            )
            return []

        active = self._get_active_slot(context_id)
        is_complete = False
        raw_overflow_word = None

        if active and active.tracker:
            if not active.tracker.word_belongs_here(word):
                next_slot = self._get_next_active_slot(active, context_id)
                word_fits_next = (
                    next_slot is not None
                    and next_slot.tracker is not None
                    and next_slot.tracker.word_belongs_here(word)
                )
                if not word_fits_next:
                    if self._streaming:
                        self._buffered_words.append(
                            _BufferedWord(word, pts, context_id, includes_inter_frame_spaces)
                        )
                        return []
                    logger.warning(
                        f"{self._name} Word '{word}' not recognised by any slot, "
                        "emitting as passthrough"
                    )
                    return [
                        self._build_word_frame(
                            word,
                            pts,
                            context_id,
                            includes_inter_frame_spaces=includes_inter_frame_spaces,
                        )
                    ]

            is_complete = active.tracker.add_word_and_check_complete(word)
            raw_overflow_word = active.tracker.get_overflow_word()
        elif self._streaming and active is None:
            self._buffered_words.append(
                _BufferedWord(word, pts, context_id, includes_inter_frame_spaces)
            )
            return []

        # Give preference to the per-call flag; fall back to the slot's flag.
        # Also propagate the per-call flag onto the slot so force_complete inherits it.
        if active and includes_inter_frame_spaces:
            active.includes_inter_frame_spaces = True
        slot_ifs = includes_inter_frame_spaces or (
            active.includes_inter_frame_spaces if active else False
        )

        frame_text = active.tracker.get_word_for_frame() if (active and active.tracker) else word
        raw_text = active.tracker.get_llm_consumed() if (active and active.tracker) else None
        suppress = active.tracker.suppress_in_context() if (active and active.tracker) else False
        emit_context_id = active.context_id if active else context_id

        frames: list[Frame] = []
        if frame_text:
            frames.append(
                self._build_word_frame(
                    frame_text,
                    pts,
                    emit_context_id,
                    raw_text=raw_text,
                    suppress_in_context=suppress,
                    includes_inter_frame_spaces=slot_ifs,
                )
            )

        if active and active.tracker and not suppress:
            frames.append(self._build_progress_frame(active, pts))

        if is_complete and active:
            active.complete = True
            frames.extend(self.flush(last_word_pts=pts))
            if raw_overflow_word:
                logger.debug(f"{self._name} Emitting overflow word '{raw_overflow_word}'")
                frames.extend(self.process_word(raw_overflow_word, pts, context_id))

        return frames

    def complete_spoken_slot(self) -> list[Frame]:
        """Mark the first pending spoken slot complete and flush unblocked skipped frames.

        Used by push_text_frames=True services: after the TTSTextFrame has been appended
        to the audio context, this marks the spoken slot done and releases any skipped
        frames waiting behind it.

        Returns:
            AggregatedTextFrame(s) that are now unblocked and should be pushed.
        """
        slot = next((s for s in self._slots if s.spoken and not s.complete), None)
        if slot:
            slot.complete = True
        return self.flush()

    def flush(self, last_word_pts: int | None = None) -> list[Frame]:
        """Walk the slot queue and return all skipped frames that are now unblocked.

        Removes complete spoken slots from the head of the queue, then emits (and
        removes) skipped slots whose preceding spoken slots are all done. Stops at
        the first incomplete spoken slot.

        Args:
            last_word_pts: When provided, skipped frames receive this PTS so they
                appear immediately after the last spoken word in the timeline.

        Returns:
            AggregatedTextFrame(s) ready to be pushed downstream.
        """
        frames: list[Frame] = []
        while self._slots:
            slot = self._slots[0]
            if slot.spoken and slot.complete:
                self._slots.pop(0)
            elif not slot.spoken and not slot.complete:
                slot.frame.append_to_context = True
                slot.frame.transport_destination = slot.transport_destination
                if last_word_pts:
                    slot.frame.pts = last_word_pts
                logger.debug(f"{self._name}: Flushing Aggregated Frame {slot.frame}")
                frames.append(slot.frame)
                slot.complete = True
                self._slots.pop(0)
            else:
                break  # spoken but not yet complete — wait
        return frames

    def force_complete(self, context_id: str, last_word_pts: int) -> list[Frame]:
        """Force-complete a context's incomplete spoken slots and flush skipped frames.

        Called at the end of an audio context to handle TTS providers that silently drop
        word-timestamp events. Emits a TTSTextFrame for any remaining unspoken text in
        each incomplete slot *belonging to this context*, marks it complete, then flushes
        all now-unblocked skipped frames. Slots for other contexts still in flight are
        left untouched so their own word events (or their own force_complete) finish them.

        The context is fully done once this returns, so its live-context and pending
        state are forgotten — any word that arrives afterwards is dropped as stale.

        Args:
            context_id: The audio context that has ended.
            last_word_pts: PTS of the last received word frame, used as the PTS for
                force-completed frames and forwarded to :meth:`flush`.

        Returns:
            Combined list of TTSTextFrames (for incomplete spoken slots) and
            AggregatedTextFrames (skipped slots now unblocked), in emission order.
        """
        frames: list[Frame] = []
        for slot in self._slots:
            if slot.spoken and not slot.complete and slot.context_id == context_id:
                if slot.tracker:
                    remaining_text = slot.tracker.get_remaining_tts_text()
                    raw_remaining = slot.tracker.get_remaining_llm_text()
                    if raw_remaining and remaining_text and remaining_text not in raw_remaining:
                        logger.warning(
                            f"{self._name} force-complete: raw_remaining {repr(raw_remaining)} "
                            f"does not contain remaining_text {repr(remaining_text)}, discarding"
                        )
                        raw_remaining = None
                    if remaining_text:
                        logger.debug(
                            f"{self._name} force-completing slot with remaining text "
                            f"{repr(remaining_text)}"
                        )
                        frames.append(
                            self._build_word_frame(
                                remaining_text,
                                last_word_pts,
                                slot.context_id,
                                raw_text=raw_remaining,
                                includes_inter_frame_spaces=slot.includes_inter_frame_spaces,
                            )
                        )
                slot.complete = True
        frames.extend(self.flush(last_word_pts=last_word_pts))
        # Context is fully done: forget it so any later word is dropped as stale.
        self._context_append_to_context.pop(context_id, None)
        self._streaming_contexts.pop(context_id, None)
        return frames

    def clear(self) -> None:
        """Clear all slots and context metadata (called on interruption/reset)."""
        self._slots.clear()
        self._context_append_to_context.clear()
        self._buffered_words.clear()
        self._streaming_contexts.clear()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _append_spoken_slot(
        self,
        frame: AggregatedTextFrame,
        context_id: str,
        tracker: WordCompletionTracker | None,
        append_to_context: bool,
        includes_inter_frame_spaces: bool,
    ) -> None:
        """Append a real, immediately-registered spoken slot.

        Shared by the non-streaming path of :meth:`register_spoken` and by
        :meth:`_promote` once a streamed sentence's boundary is confirmed.
        """
        self._context_append_to_context[context_id] = append_to_context
        self._slots.append(
            _AggregatedFrameSlot(
                frame=frame,
                context_id=context_id,
                spoken=True,
                tracker=tracker,
                includes_inter_frame_spaces=includes_inter_frame_spaces,
            )
        )

    def _promote(
        self, agg: _ParallelAggregation, context_id: str, append_to_context: bool
    ) -> list[Frame]:
        """Turn a completed parallel-aggregated sentence into a real spoken slot.

        Builds the real WordCompletionTracker and an ``AggregationType.SENTENCE``
        AggregatedTextFrame from the three aggregated text channels, appends the
        slot for ``context_id``, then replays any words that were buffered waiting
        for it.

        The sentence frame is **emitted downstream** (first in the returned list)
        with ``will_be_spoken=True``: it is the streaming-mode equivalent of the
        pre-synthesis sentence frame that SENTENCE mode pushes, giving RTVI clients
        the initial ``spoken_status="new"`` event whose ``segment_id`` the
        subsequent progress frames (built from this same frame's id) reference.
        ``append_to_context`` is False on it — the conversation context is built
        from the per-word TTSTextFrames, not this announcement.

        The slot's ``includes_inter_frame_spaces`` is left False: for a streamed
        (TOKEN-mode) sentence, per-word CJK spacing is stamped by
        :meth:`process_word` from ``add_word_timestamps``, never from the incoming
        LLM token's own inter-frame-space flag.

        Args:
            agg: The completed sentence, in the three parallel text channels.
            context_id: The context the promoted slot belongs to.
            append_to_context: Whether word frames for the slot append to context.

        Returns:
            The sentence frame followed by any frames unblocked by replaying
            previously-buffered words. Empty if the aggregated text is entirely
            whitespace.
        """
        if not agg.user_facing_text.strip():
            return []

        frame = AggregatedTextFrame(
            agg.user_facing_text, AggregationType.SENTENCE, raw_text=agg.llm_text or None
        )
        frame.context_id = context_id
        frame.will_be_spoken = True
        frame.append_to_context = False
        tracker = WordCompletionTracker(
            agg.tts_text, llm_text=agg.llm_text or None, user_facing_text=agg.user_facing_text
        )
        self._append_spoken_slot(frame, context_id, tracker, append_to_context, False)
        return [frame, *self._drain_buffered_words()]

    def _drain_buffered_words(self) -> list[Frame]:
        """Replay previously-buffered word events now that a new slot may match them.

        Snapshots and clears the buffer before replaying so a word that still
        doesn't match anything (it belongs to a later, still-pending sentence)
        gets re-buffered by :meth:`process_word` itself and waits for the next
        promotion, rather than looping here.
        """
        buffered = self._buffered_words
        self._buffered_words = []
        frames: list[Frame] = []
        for w in buffered:
            frames.extend(
                self.process_word(w.word, w.pts, w.context_id, w.includes_inter_frame_spaces)
            )
        return frames

    def _slot_matches_context(self, slot: _AggregatedFrameSlot, context_id: str | None) -> bool:
        """Whether *slot* is an eligible active slot for *context_id*.

        A ``None`` context_id (legacy providers with no per-context word tagging, where
        concurrency can't occur) matches any slot, preserving the untargeted behaviour.
        Otherwise word routing is scoped to the slot's own context so concurrently-live
        contexts never consume each other's words.
        """
        if not (slot.spoken and not slot.complete and slot.tracker is not None):
            return False
        return context_id is None or slot.context_id == context_id

    def _get_active_slot(self, context_id: str | None = None) -> _AggregatedFrameSlot | None:
        """Return the first incomplete spoken slot with a tracker for *context_id*."""
        return next(
            (s for s in self._slots if self._slot_matches_context(s, context_id)),
            None,
        )

    def _get_next_active_slot(
        self, current: _AggregatedFrameSlot, context_id: str | None = None
    ) -> _AggregatedFrameSlot | None:
        """Return the first incomplete spoken slot with a tracker for *context_id* after *current*."""
        found = False
        for s in self._slots:
            if s is current:
                found = True
                continue
            if found and self._slot_matches_context(s, context_id):
                return s
        return None

    def _build_progress_frame(
        self, slot: _AggregatedFrameSlot, pts: int
    ) -> AggregatedTextProgressFrame:
        """Build an AggregatedTextProgressFrame reflecting the current spoken/remaining state of a slot."""
        assert slot.tracker is not None
        frame = AggregatedTextProgressFrame(
            segment_id=slot.frame.id,
            context_id=slot.context_id,
            text=slot.frame.text,
            aggregated_by=slot.frame.aggregated_by,
            accumulated_text=slot.tracker.get_accumulated_user_facing_text(),
            remaining_text=slot.tracker.get_remaining_user_facing_text(strip=False),
        )
        frame.pts = pts
        return frame

    def _build_word_frame(
        self,
        text: str,
        pts: int,
        context_id: str | None,
        raw_text: str | None = None,
        suppress_in_context: bool = False,
        includes_inter_frame_spaces: bool = False,
    ) -> Frame:
        """Build a TTSTextFrame with all standard word-timestamp attributes set."""
        frame = TTSTextFrame(text, aggregated_by=AggregationType.WORD)
        frame.pts = pts
        frame.context_id = context_id
        if suppress_in_context:
            frame.append_to_context = False
        else:
            frame.append_to_context = (
                self._context_append_to_context.get(context_id, True)
                if context_id is not None
                else True
            )
        frame.raw_text = raw_text
        frame.includes_inter_frame_spaces = includes_inter_frame_spaces
        return frame
