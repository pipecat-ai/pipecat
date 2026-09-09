#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decision engine holding a speculative bot response until its turn is confirmed."""

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum

from loguru import logger

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    SystemFrame,
    UninterruptibleFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.base_object import BaseObject
from pipecat.utils.frame_queue import FrameQueue

# A frame paired with the direction it travels in.
GatedFrame = tuple[Frame, FrameDirection]


class SpeculationState(Enum):
    """What the gate is doing with the frames passing through it.

    - ``OPEN``: emitting everything.
    - ``HOLDING``: holding a speculative response until it is confirmed.
    - ``DROPPING``: discarding the rest of a withdrawn speculative response.
    """

    OPEN = "open"
    HOLDING = "holding"
    DROPPING = "dropping"


class SpeculationGate(BaseObject):
    """Decides which frames of a speculative bot response may be emitted, and when.

    A speculative response is generated from an eager end of turn — a
    provisional guess that the user has finished talking — so it may answer a
    transcript the user never actually completed. This holds everything such a
    response produces until the turn is confirmed, then releases it, or discards
    it if the guess is withdrawn.

    The response is bounded by
    :class:`~pipecat.frames.frames.LLMFullResponseStartFrame` and
    :class:`~pipecat.frames.frames.LLMFullResponseEndFrame`, which the LLM
    service stamps with a ``speculation_id``. A
    :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` naming that id
    releases the response (the turn ended, and this response answers it) and an
    :class:`~pipecat.frames.frames.EagerEndOfTurnCancelFrame` discards it.

    Only one speculation is ever in flight, since producing one takes a whole
    user turn, so the gate tracks a single response. Both signals still carry an
    id: a turn that ends without naming a speculation must not release one, and
    a confirmation can arrive before the response it confirms, since Pipecat
    dispatches system frames ahead of the queued frames they pass.

    While holding, everything is held in arrival order except system frames,
    which are out-of-band throughout Pipecat — and which carry the verdicts the
    gate is waiting for, so holding them would deadlock it. That includes
    :class:`~pipecat.frames.frames.UninterruptibleFrame` ones, which are ordered
    like any other frame; discarding a speculation keeps them and emits them on,
    since they can belong to work started before it.

    This decides rather than processes frames: :meth:`process` is synchronous
    and returns the frames its caller should push, in order. A host can
    therefore push from several tasks at once — every state transition
    completes without an await for another task to interleave with.
    :class:`~pipecat.services.llm_service.LLMService` hosts one, gating every
    response it produces.

    A hold bounds itself. A service that stops sending turn signals
    mid-speculation would otherwise leave the bot silent for the rest of the
    session, so a hold that outlasts ``max_hold_duration`` is discarded. That
    is the one point where frames leave with no caller to hand them to, so
    they go to ``push_expired`` instead of being returned.

    Example::

        gate = SpeculationGate(push_expired=self._push_past_gate)
        await gate.setup(task_manager)

        for frame, direction in gate.process(frame, direction):
            await self.push_frame(frame, direction)
    """

    def __init__(
        self,
        *,
        push_expired: Callable[[list[GatedFrame]], Awaitable[None]],
        max_hold_duration: float = 5.0,
        **kwargs,
    ):
        """Initialize the speculation gate.

        Args:
            push_expired: Pushes the frames a hold leaves behind when it
                outlasts ``max_hold_duration``: whatever the discarded response
                was holding back that has to be delivered anyway. These are
                handed over rather than returned because the hold runs out on
                the gate's own task, with no :meth:`process` caller waiting to
                push them.
            max_hold_duration: Seconds a response may be held before the gate
                gives up on it.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._push_expired = push_expired
        self._max_hold_duration = max_hold_duration
        # A hold is bounded by a task waiting for it to end. Waiting rather than
        # being cancelled at the end: holds start and end inside `process`,
        # which is synchronous, and cancelling a task has to be awaited.
        self._hold_ended = asyncio.Event()
        self._hold_timeout_task: asyncio.Task | None = None
        self._state = SpeculationState.OPEN
        self._speculation_id: str | None = None
        self._buffer: FrameQueue = FrameQueue(frame_getter=lambda item: item[0])
        # A speculation confirmed before its response arrived. The confirmation
        # travels as a system frame, so it can pass the queued frames it
        # confirms, and nothing follows it to correct a response held by
        # mistake — the turn is over, so no further response is coming.
        # One slot is enough: only one speculation is ever in flight.
        self._confirmed_id: str | None = None

    async def cleanup(self):
        """Stop bounding a hold that outlived the pipeline."""
        await super().cleanup()
        if self._hold_timeout_task:
            task, self._hold_timeout_task = self._hold_timeout_task, None
            await self.cancel_task(task)

    @property
    def state(self) -> SpeculationState:
        """What the gate is currently doing with frames passing through it."""
        return self._state

    @property
    def speculation_id(self) -> str | None:
        """The speculation being held, or None when nothing is held."""
        return self._speculation_id if self._state == SpeculationState.HOLDING else None

    def process(self, frame: Frame, direction: FrameDirection) -> list[GatedFrame]:
        """Decide what a frame passing through the gate releases.

        Args:
            frame: The frame to process.
            direction: The direction the frame travels in.

        Returns:
            The frames to push, in order. Empty while a frame is held or
            dropped; longer than one frame when a verdict releases what was
            held behind it.
        """
        if direction != FrameDirection.DOWNSTREAM:
            return [(frame, direction)]

        if isinstance(frame, SystemFrame):
            # Emitted before the verdict is applied, so a released response
            # still follows the frame that ended the turn it answers.
            emitted = [(frame, direction)]
            if isinstance(frame, EagerEndOfTurnCancelFrame):
                emitted += self._discard(frame.speculation_id)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                emitted += self._release(frame.speculation_id)
            elif isinstance(frame, InterruptionFrame):
                emitted += self._discard(None)
            return emitted

        if isinstance(frame, EndFrame):
            # Uninterruptible, and the runner awaits it: holding it hangs
            # shutdown. Discarding first delivers whatever is held that has to
            # outlive the speculation, ahead of it.
            return self._discard(None) + [(frame, direction)]

        emitted: list[GatedFrame] = []
        if isinstance(frame, LLMFullResponseStartFrame):
            emitted += self._begin(frame.speculation_id)

        if self._state == SpeculationState.HOLDING:
            # Held in arrival order, uninterruptible frames included: they are
            # ordered like any other, and the buffer preserves them when the
            # speculation around them is discarded.
            self._buffer.put_nowait((frame, direction))
        elif self._state == SpeculationState.DROPPING:
            if isinstance(frame, UninterruptibleFrame):
                # Not part of the response being dropped, and nothing is being
                # held back, so emitting it keeps it in order.
                emitted.append((frame, direction))
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._state = SpeculationState.OPEN
        else:
            emitted.append(self._resolved(frame, direction))

        return emitted

    def _bound_hold(self):
        """Start the clock on the hold just taken.

        Starting a task needs no await, so this runs inside the state
        transition that took the hold. A hold that supersedes another leaves
        that one's task to notice the end and exit.
        """
        self._hold_ended.clear()
        self._hold_timeout_task = self.create_task(self._hold_timeout_handler(), "_hold_timeout")

    def _end_hold(self, state: SpeculationState):
        """Leave the hold, releasing the bound on it.

        The single exit from ``HOLDING``, which is what lets the bound end
        exactly once however the hold ended.

        Args:
            state: What the gate does with the frames that follow.
        """
        self._state = state
        self._speculation_id = None
        self._hold_ended.set()

    async def _hold_timeout_handler(self):
        """Discard a hold that outlasted its bound."""
        try:
            await asyncio.wait_for(self._hold_ended.wait(), self._max_hold_duration)
            return
        except TimeoutError:
            pass

        logger.warning(
            f"{self}: speculative response unresolved after {self._max_hold_duration}s, "
            "discarding it"
        )
        await self._push_expired(self._discard(None))

    def _resolved(self, frame: Frame, direction: FrameDirection) -> GatedFrame:
        """Prepare a frame the gate has resolved.

        A response that gets past the gate has been confirmed, or never belonged
        to a speculation, so it carries no speculation id onward. That keeps the
        id meaning exactly one thing downstream: this response answers a turn
        that may not have ended.
        """
        if isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
            frame.speculation_id = None
        return (frame, direction)

    def _begin(self, speculation_id: str | None) -> list[GatedFrame]:
        """Decide what to do with the response this id opens."""
        emitted: list[GatedFrame] = []

        if self._state != SpeculationState.OPEN:
            # A new response supersedes the one we were holding or dropping. A
            # withdrawn response may never send its end frame, since its
            # generation was cancelled mid-flight, and an unconfirmed one is
            # void once something else starts answering. Nothing of it can still
            # be queued behind this frame, so there is no tail left to drop.
            emitted += self._drop_held("superseded by a new response", keep_dropping=False)

        if not speculation_id:
            return emitted

        if speculation_id == self._confirmed_id:
            # Confirmed before it reached us; nothing left to hold back.
            self._confirmed_id = None
            return emitted

        self._state = SpeculationState.HOLDING
        self._speculation_id = speculation_id
        self._bound_hold()
        return emitted

    def _release(self, speculation_id: str | None) -> list[GatedFrame]:
        """Release the response the turn end confirms.

        Args:
            speculation_id: The speculation the turn confirms. A turn that ends
                without naming one confirms nothing, and releases nothing.

        Returns:
            The frames the confirmation releases, in arrival order.
        """
        if not speculation_id:
            return []

        if self._state != SpeculationState.HOLDING or speculation_id != self._speculation_id:
            # Confirmed before its response reached us. Remember it, or the
            # response would be held on arrival and never released: the turn is
            # over, so nothing follows to supersede it.
            self._confirmed_id = speculation_id
            return []

        logger.debug(f"{self}: releasing speculative response ({self._buffer.qsize()} frames)")
        self._end_hold(SpeculationState.OPEN)
        return self._flush()

    def _discard(self, speculation_id: str | None) -> list[GatedFrame]:
        """Discard the response a withdrawal or an interruption voids.

        A withdrawal that arrives before the response it voids needs no memory:
        the response is held on arrival, and whatever answers the turn instead
        supersedes it — or the bound on the hold runs out if nothing does.

        Args:
            speculation_id: The speculation being withdrawn, or None to discard
                whatever is held, which is what an interruption and shutdown do.

        Returns:
            Whatever the discarded response was holding back that has to be
            delivered anyway.
        """
        if speculation_id and speculation_id != self._speculation_id:
            return []

        return self._drop_held("withdrawn", keep_dropping=True)

    def _drop_held(self, reason: str, *, keep_dropping: bool) -> list[GatedFrame]:
        """Drop the held response.

        Args:
            reason: Why it is being dropped, for the log line.
            keep_dropping: Whether the rest of the response may still be queued
                behind us and has to be dropped as it arrives. False when
                something already past it proves there is no tail left.

        Returns:
            The held frames that must be delivered anyway, in arrival order.
        """
        if self._state != SpeculationState.HOLDING:
            self._state = SpeculationState.OPEN
            self._speculation_id = None
            return []

        logger.debug(
            f"{self}: discarding speculative response ({self._buffer.qsize()} frames, {reason})"
        )
        # A response that already ended has no tail left to drop.
        complete = self._buffer.has_frame(LLMFullResponseEndFrame)
        # Drops the speculative response and keeps anything that must always be
        # delivered, which is then emitted rather than discarded with it.
        self._buffer.reset()
        self._end_hold(
            SpeculationState.DROPPING if keep_dropping and not complete else SpeculationState.OPEN
        )
        return self._flush()

    def _flush(self) -> list[GatedFrame]:
        """Take everything the buffer still holds, in the order it arrived."""
        emitted = []
        while not self._buffer.empty():
            frame, direction = self._buffer.get_nowait()
            self._buffer.task_done()
            emitted.append(self._resolved(frame, direction))
        return emitted
