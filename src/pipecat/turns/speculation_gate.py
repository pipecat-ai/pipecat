#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decision engine holding a speculative bot response until its turn ends."""

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
    UserStartedSpeakingFrame,
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
    - ``HOLDING``: holding a speculative response until its turn ends.
    - ``DROPPING``: discarding the rest of a withdrawn speculative response.
    """

    OPEN = "open"
    HOLDING = "holding"
    DROPPING = "dropping"


class SpeculationGate(BaseObject):
    """Decides which frames of a speculative bot response may be emitted, and when.

    A speculative response is generated from an eager end of turn, a
    provisional guess that the user has finished talking, so it may answer a
    transcript the user never actually completed. This holds everything such a
    response produces while the user turn is still open, then releases it when
    the turn ends, or discards it if the guess is withdrawn.

    A host calls :meth:`begin_speculation` when it starts an inference, saying
    whether the turn it answers may not have ended. A speculative response is
    held from its :class:`~pipecat.frames.frames.LLMFullResponseStartFrame`
    onward. A :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` ends the
    turn and releases the response, and an
    :class:`~pipecat.frames.frames.EagerEndOfTurnCancelFrame` withdraws the
    speculation and discards it.

    There is one speculation per user turn and one component that settles it,
    the stop strategy that started it, so the gate needs no identity for it:
    the turn's end releases whatever is held, and a withdrawal discards it.
    The two are system frames, which keep their order, so a withdrawal always
    precedes the turn end that follows it. What the gate does track is whether
    the turn is open, from the user speaking frames: the turn end is a system
    frame and can overtake the queued context frame that starts the inference
    it confirms, and a speculative inference that begins after its turn has
    ended answers a turn that is over, so it is not held at all.

    While holding, everything is held in arrival order except system frames,
    which are out-of-band throughout Pipecat, and which carry the verdicts the
    gate is waiting for, so holding them would deadlock it. That includes
    :class:`~pipecat.frames.frames.UninterruptibleFrame` ones, which are ordered
    like any other frame; discarding a speculation keeps them and emits them on,
    since they can belong to work started before it.

    This decides rather than processes frames: :meth:`process` is synchronous
    and returns the frames its caller should push, in order. A host can
    therefore push from several tasks at once, since every state transition
    completes without an await for another task to interleave with.
    :class:`~pipecat.services.llm_service.LLMService` hosts one, gating every
    response it produces.

    Example::

        gate = SpeculationGate()
        for frame, direction in gate.process(frame, direction):
            await self.push_frame(frame, direction)
    """

    def __init__(self, **kwargs):
        """Initialize the speculation gate.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._state = SpeculationState.OPEN
        # Whether the inference in flight answers a turn that has not ended. Set
        # when the inference starts, cleared when the turn settles it either
        # way, so it outlives any one response frame.
        self._speculating = False
        # Whether the user turn has ended and no new one has begun. A
        # speculative inference that begins in that state answers a turn that
        # is already over, so its response needs no holding.
        self._turn_ended = False
        self._buffer: FrameQueue = FrameQueue(frame_getter=lambda item: item[0])

    @property
    def state(self) -> SpeculationState:
        """What the gate is currently doing with frames passing through it."""
        return self._state

    @property
    def speculating(self) -> bool:
        """Whether the inference in flight answers a turn that has not ended.

        Set from :meth:`begin_speculation` until the turn settles it, so it
        covers the whole inference rather than only the stretch with frames in
        flight. False means whatever is generating now answers a turn that has
        ended, and its side effects can be let through. Use :attr:`state` to ask
        the narrower question of whether frames are being held right now.
        """
        return self._speculating

    def begin_speculation(self, speculative: bool):
        """Take note of an inference starting, and whether its turn may not have ended.

        Called when the inference starts rather than when its first response
        frame arrives, so a turn that ended in between is already accounted for
        by the time the response shows up.

        Args:
            speculative: Whether the inference answers an eager end of turn.
                False for an ordinary inference against a turn that has ended.
        """
        # A speculative inference that begins after its turn ended answers a
        # turn that is over: the turn end overtook its context frame.
        self._speculating = speculative and not self._turn_ended

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
            if isinstance(frame, UserStartedSpeakingFrame):
                self._turn_ended = False
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._turn_ended = True
                emitted += self._release()
            elif isinstance(frame, (EagerEndOfTurnCancelFrame, InterruptionFrame)):
                emitted += self._discard()
            return emitted

        if isinstance(frame, EndFrame):
            # Uninterruptible, and the runner awaits it: holding it hangs
            # shutdown. Discarding first delivers whatever is held that has to
            # outlive the speculation, ahead of it.
            return self._discard() + [(frame, direction)]

        emitted: list[GatedFrame] = []
        if isinstance(frame, LLMFullResponseStartFrame):
            emitted += self._begin()

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
            emitted.append((frame, direction))
        return emitted

    def _begin(self) -> list[GatedFrame]:
        """Decide what to do with the response opening here."""
        emitted: list[GatedFrame] = []
        if self._state != SpeculationState.OPEN:
            # A new response supersedes the one we were holding or dropping. A
            # withdrawn response may never send its end frame, since its
            # generation was cancelled mid-flight, and an unconfirmed one is
            # void once something else starts answering. Nothing of it can still
            # be queued behind this frame, so there is no tail left to drop.
            emitted += self._drop_held("superseded by a new response", keep_dropping=False)
        # Whether this response is speculative was settled when its inference
        # started, so a turn that ended since then already cleared it.
        if self._speculating:
            self._state = SpeculationState.HOLDING
        return emitted

    def _release(self) -> list[GatedFrame]:
        """Release the held response: the turn it answers has ended.

        Returns:
            The frames the turn end releases, in arrival order.
        """
        # The turn ended, so nothing this inference still produces is
        # speculative, including a tool call it has yet to reach.
        self._speculating = False
        if self._state != SpeculationState.HOLDING:
            return []
        logger.debug(f"{self}: releasing speculative response ({self._buffer.qsize()} frames)")
        self._state = SpeculationState.OPEN
        return self._flush()

    def _discard(self) -> list[GatedFrame]:
        """Discard the held response: the speculation was withdrawn, or an interruption voided it.

        A withdrawal that arrives before the response it voids needs no memory:
        the response is held on arrival, and whatever answers the turn instead
        supersedes it.

        Returns:
            Whatever the discarded response was holding back that has to be
            delivered anyway.
        """
        # The speculation is void, so nothing the inference still produces
        # answers a turn worth holding for.
        self._speculating = False
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
            return []
        logger.debug(
            f"{self}: discarding speculative response ({self._buffer.qsize()} frames, {reason})"
        )
        # A response that already ended has no tail left to drop.
        complete = self._buffer.has_frame(LLMFullResponseEndFrame)
        # Drops the speculative response and keeps anything that must always be
        # delivered, which is then emitted rather than discarded with it.
        self._buffer.reset()
        self._state = (
            SpeculationState.DROPPING if keep_dropping and not complete else SpeculationState.OPEN
        )
        return self._flush()

    def _flush(self) -> list[GatedFrame]:
        """Take everything the buffer still holds, in the order it arrived."""
        emitted = []
        while not self._buffer.empty():
            frame, direction = self._buffer.get_nowait()
            self._buffer.task_done()
            emitted.append((frame, direction))
        return emitted
