#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ordered sequencer for AggregatedTextFrame slots through TTS processing."""

from dataclasses import dataclass, field

from loguru import logger

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregationType,
    Frame,
    TTSTextFrame,
)
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker


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


class AggregatedFrameSequencer:
    """Sequences AggregatedTextFrame slots to preserve TTS context ordering.

    Manages an ordered queue of spoken and skipped TTS slots. Spoken slots are tracked
    via a :class:`WordCompletionTracker`; skipped slots (e.g. code blocks excluded from
    TTS synthesis) wait in-place until all preceding spoken slots are complete, then are
    flushed downstream with ``append_to_context=True``.

    All methods are synchronous and return lists of frames the caller should push
    downstream, making the sequencer fully testable without any async machinery.

    Example::

        sequencer = AggregatedFrameSequencer()
        sequencer.register_spoken(frame, ctx_id, tracker, append_to_context=True)
        for f in sequencer.process_word("hello", pts=1000, context_id=ctx_id):
            await self.push_frame(f)
    """

    def __init__(self, name: str = "AggregatedFrameSequencer"):
        """Initialize the sequencer.

        Args:
            name: Label used in log messages (typically the owning TTS service name).
        """
        self._name = name
        self._slots: list[_AggregatedFrameSlot] = []
        self._context_append_to_context: dict[str, bool] = {}

    def register_spoken(
        self,
        frame: AggregatedTextFrame,
        context_id: str,
        tracker: WordCompletionTracker | None,
        append_to_context: bool,
    ) -> None:
        """Register a spoken AggregatedTextFrame slot.

        Called from _push_tts_frames for frames sent to the TTS service. The slot is
        marked complete either via :meth:`process_word` (word-timestamp services) or
        :meth:`complete_spoken_slot` (push_text_frames=True services).

        Args:
            frame: The AggregatedTextFrame being spoken.
            context_id: The TTS context ID assigned to this frame.
            tracker: WordCompletionTracker for word-timestamp services; None for
                push_text_frames=True services (they complete via complete_spoken_slot).
            append_to_context: Whether word frames built for this context should carry
                append_to_context=True.
        """
        self._context_append_to_context[context_id] = append_to_context
        self._slots.append(
            _AggregatedFrameSlot(
                frame=frame,
                context_id=context_id,
                spoken=True,
                tracker=tracker,
            )
        )

    def register_skipped(
        self,
        frame: AggregatedTextFrame,
        context_id: str,
        transport_destination: str | None,
    ) -> list[Frame]:
        """Register a skipped AggregatedTextFrame and attempt an immediate flush.

        The frame is appended as a skipped slot. If no incomplete spoken slot precedes
        it, the frame is returned right away; otherwise it waits until a later
        :meth:`flush` unblocks it.

        Args:
            frame: The skipped AggregatedTextFrame (e.g. a code block).
            context_id: The context ID assigned in _push_tts_frames.
            transport_destination: Transport routing value to attach at flush time.

        Returns:
            Frames to push downstream (empty when blocked by a preceding spoken slot).
        """
        frame.context_id = context_id
        self._slots.append(
            _AggregatedFrameSlot(
                frame=frame,
                context_id=context_id,
                spoken=False,
                transport_destination=transport_destination,
            )
        )
        return self.flush()

    def process_word(
        self,
        word: str,
        pts: int,
        context_id: str | None,
    ) -> list[Frame]:
        """Process one word-timestamp event and return frames to push downstream.

        Locates the active (first incomplete spoken) slot with a tracker, advances it
        by the incoming word, and builds a :class:`TTSTextFrame`. Handles:

        - Normal words that fit entirely within the active slot.
        - Overflow words straddling two slot boundaries.
        - Force-complete when the TTS drops an event (word belongs to the next slot).
        - Passthrough for words not recognised by any slot.
        - Flushes any skipped slots unblocked by slot completion.

        Args:
            word: A word token from the TTS service word-timestamp stream.
            pts: Presentation timestamp (nanoseconds) to assign to the frame.
            context_id: TTS context ID from the word-timestamp event.

        Returns:
            Ordered list of frames (TTSTextFrame and/or AggregatedTextFrame) to push.
        """
        active = self._get_active_slot()
        is_complete = False
        raw_overflow_word = None

        if active and active.tracker:
            if not active.tracker.word_belongs_here(word):
                next_slot = self._get_next_active_slot(active)
                word_fits_next = (
                    next_slot is not None
                    and next_slot.tracker is not None
                    and next_slot.tracker.word_belongs_here(word)
                )
                if not word_fits_next:
                    logger.warning(
                        f"{self._name} Word '{word}' not recognised by any slot, "
                        "emitting as passthrough"
                    )
                    return [self._build_word_frame(word, pts, context_id)]

            is_complete = active.tracker.add_word_and_check_complete(word)
            raw_overflow_word = active.tracker.get_overflow_word()

        frame_text = active.tracker.get_word_for_frame() if (active and active.tracker) else word
        raw_text = active.tracker.get_llm_consumed() if (active and active.tracker) else None
        emit_context_id = active.context_id if active else context_id

        logger.debug(f"{self._name} Word '{word}' → frame_text='{frame_text}', raw='{raw_text}'")
        frames: list[Frame] = [
            self._build_word_frame(frame_text, pts, emit_context_id, raw_text=raw_text)
        ]

        if is_complete:
            active.complete = True
            frames.extend(self.flush(last_word_pts=pts))
            if raw_overflow_word:
                logger.debug(f"{self._name} Emitting overflow word '{raw_overflow_word}'")
                frames.extend(self._process_overflow(raw_overflow_word, pts))

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

    def force_complete(self, last_word_pts: int) -> list[Frame]:
        """Force-complete all incomplete spoken slots and flush skipped frames.

        Called at the end of an audio context to handle TTS providers that silently drop
        word-timestamp events. Emits a TTSTextFrame for any remaining unspoken text in
        each incomplete slot, marks it complete, then flushes all now-unblocked skipped
        frames.

        Args:
            last_word_pts: PTS of the last received word frame, used as the PTS for
                force-completed frames and forwarded to :meth:`flush`.

        Returns:
            Combined list of TTSTextFrames (for incomplete spoken slots) and
            AggregatedTextFrames (skipped slots now unblocked), in emission order.
        """
        frames: list[Frame] = []
        for slot in self._slots:
            if slot.spoken and not slot.complete:
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
                            )
                        )
                slot.complete = True
        frames.extend(self.flush(last_word_pts=last_word_pts))
        return frames

    def clear(self) -> None:
        """Clear all slots and context metadata (called on interruption/reset)."""
        self._slots.clear()
        self._context_append_to_context.clear()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_active_slot(self) -> _AggregatedFrameSlot | None:
        """Return the first incomplete spoken slot that has a tracker."""
        return next(
            (s for s in self._slots if s.spoken and not s.complete and s.tracker is not None),
            None,
        )

    def _get_next_active_slot(self, current: _AggregatedFrameSlot) -> _AggregatedFrameSlot | None:
        """Return the first incomplete spoken slot with a tracker after *current*."""
        found = False
        for s in self._slots:
            if s is current:
                found = True
                continue
            if found and s.spoken and not s.complete and s.tracker is not None:
                return s
        return None

    def _build_word_frame(
        self,
        text: str,
        pts: int,
        context_id: str | None,
        raw_text: str | None = None,
    ) -> TTSTextFrame:
        """Build a TTSTextFrame with all standard word-timestamp attributes set."""
        frame = TTSTextFrame(text, aggregated_by=AggregationType.WORD)
        frame.pts = pts
        frame.context_id = context_id
        frame.append_to_context = self._context_append_to_context.get(context_id, True)
        frame.raw_text = raw_text
        return frame

    def _process_overflow(self, raw_overflow_word: str, pts: int) -> list[Frame]:
        """Feed an overflow suffix into the next active slot and return resulting frames."""
        frames: list[Frame] = []
        next_active = self._get_active_slot()
        if not next_active or not next_active.tracker:
            return frames
        overflow_complete = next_active.tracker.add_word_and_check_complete(raw_overflow_word)
        frames.append(
            self._build_word_frame(
                raw_overflow_word,
                pts,
                next_active.context_id,
                raw_text=next_active.tracker.get_llm_consumed(),
            )
        )
        if overflow_complete:
            next_active.complete = True
            frames.extend(self.flush(last_word_pts=pts))
        return frames
