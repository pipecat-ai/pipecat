#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decision engine holding a speculative bot response until its turn is confirmed."""

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


class SpeculationGate:
    """Decides which frames of a speculative bot response may be emitted, and when.

    A speculative response is generated from an eager end of turn — a
    provisional guess that the user has finished talking — so it may answer a
    transcript the user never actually completed. This holds everything such a
    response produces until the turn is confirmed, then releases it, or discards
    it if the guess is withdrawn.

    A host calls :meth:`begin_speculation` when it starts an inference for a
    turn that may not have ended, and the response that follows is held from its
    :class:`~pipecat.frames.frames.LLMFullResponseStartFrame` onward. A
    :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` releases it — the
    turn it answers has ended — and an
    :class:`~pipecat.frames.frames.EagerEndOfTurnCancelFrame` discards it.
    Neither names a response: only one speculation is ever in flight, since
    producing one takes a whole user turn.

    Whether an inference is still answering an unconfirmed turn is therefore the
    gate's to answer, through :attr:`is_speculating`, rather than something a
    host tracks alongside it. A host that kept its own copy would have to clear
    it on every path that settles a speculation, and the two would disagree on
    the paths it missed.

    The gate also tracks whether the turn has ended, because a confirmation can
    arrive before the inference it confirms — Pipecat dispatches system frames
    ahead of the queued frames they pass — and a response for a turn already
    over is not held at all.

    Both of those resolving frames reach the gate ahead of anything queued
    behind them, which is what keeps a withdrawal in front of the turn end that
    follows it. A withdrawal arriving second would find the response already
    released.

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

    Holding is unbounded here. Whoever started the speculation bounds it and
    withdraws it if it goes unresolved — see
    :class:`~pipecat.turns.user_stop.EagerUserTurnStopStrategy` — so every way a
    speculation ends reaches this gate as a frame, and the gate never has to
    abandon one on its own.

    Example::

        for frame, direction in gate.process(frame, direction):
            await self.push_frame(frame, direction)
    """

    def __init__(self, *, name: str = "SpeculationGate"):
        """Initialize the speculation gate.

        Args:
            name: Label used in log messages, typically the owning service's.
        """
        self._name = name
        self._state = SpeculationState.OPEN
        # Whether the inference in flight answers a turn that has not been
        # confirmed. Set when the inference starts, cleared when the turn
        # settles it either way — so it outlives any one response frame.
        self._speculating = False
        # Whether the turn ended since it was last seen to start. A speculation
        # registered while this is set answers a turn that is already over, so
        # it is not held. Starts False, so a gate that never sees turn frames
        # holds rather than speaks — the strategy's bound then recovers it.
        self._turn_ended = False
        self._buffer: FrameQueue = FrameQueue(frame_getter=lambda item: item[0])

    def __str__(self):
        return self._name

    @property
    def state(self) -> SpeculationState:
        """What the gate is currently doing with frames passing through it."""
        return self._state

    @property
    def is_speculating(self) -> bool:
        """Whether the inference in flight answers an unconfirmed turn.

        Set from :meth:`begin_speculation` until the turn settles it, so it
        covers the whole inference rather than only the stretch with frames in
        flight. False means whatever is generating now answers a turn that has
        ended, and its side effects can be let through. Use :attr:`state` to ask
        the narrower question of whether frames are being held right now.
        """
        return self._speculating

    def begin_speculation(self, speculation: bool):
        """Take note of an inference whose turn may not have ended.

        Called when the inference starts rather than when its first response
        frame arrives, so a turn confirmed in between is already accounted for
        by the time the response shows up.

        Args:
            speculation: Whether the inference answers a turn that may still be
                open, rather than an ordinary one against a turn that has ended.
        """
        # A turn that ended before the inference reached us leaves nothing to
        # hold for: the response answers a turn that is already over.
        self._speculating = speculation and not self._turn_ended

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
            if isinstance(frame, (EagerEndOfTurnCancelFrame, InterruptionFrame)):
                emitted += self._discard()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                emitted += self._release()
            elif isinstance(frame, UserStartedSpeakingFrame):
                # A turn is open again, so an inference started from here on
                # answers something that may not have ended.
                self._turn_ended = False
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

    def _end_hold(self, state: SpeculationState):
        """Leave the hold.

        The single exit from ``HOLDING``. Whether the speculation itself is over
        is a separate question — a response superseded by a new one ends its
        hold while the inference that replaced it is still pending.

        Args:
            state: What the gate does with the frames that follow.
        """
        self._state = state

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
        # started, so a turn confirmed since then already cleared it.
        if not self._speculating:
            return emitted

        self._state = SpeculationState.HOLDING
        return emitted

    def _release(self) -> list[GatedFrame]:
        """Release whatever the turn ending confirms.

        Returns:
            The frames the confirmation releases, in arrival order.
        """
        # Remembered so an inference that arrives after this is not held: the
        # turn it answers is already over, and nothing follows to release it.
        self._turn_ended = True

        # Nothing this inference still produces is speculative either —
        # including a tool call it has yet to reach.
        self._speculating = False

        if self._state != SpeculationState.HOLDING:
            return []

        logger.debug(f"{self}: releasing speculative response ({self._buffer.qsize()} frames)")
        self._end_hold(SpeculationState.OPEN)
        return self._flush()

    def _discard(self) -> list[GatedFrame]:
        """Discard the response a withdrawal or an interruption voids.

        A withdrawal that arrives before the response it voids needs no memory:
        the response is held on arrival, and whatever answers the turn instead
        supersedes it — or the bound on the hold runs out if nothing does.

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
            emitted.append((frame, direction))
        return emitted
