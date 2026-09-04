#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gate that holds a speculative bot response until the turn it answers is confirmed."""

import asyncio
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
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class SpeculationState(Enum):
    """What the gate is doing with the frames passing through it.

    - ``OPEN``: forwarding everything.
    - ``HOLDING``: holding a speculative response until it is confirmed.
    - ``DROPPING``: discarding the rest of a withdrawn speculative response.
    """

    OPEN = "open"
    HOLDING = "holding"
    DROPPING = "dropping"


class SpeculativeResponseGate(FrameProcessor):
    """Holds a speculative bot response until the user turn it answers is confirmed.

    A speculative response is generated from an eager end of turn — a
    provisional guess that the user has finished talking — so it may answer a
    transcript the user never actually completed. This gate holds everything
    such a response produces until the turn is confirmed, then releases it, or
    discards it if the guess is withdrawn.

    Place it anywhere before the output transport, which is the point where
    unconfirmed speech would reach the user::

        [llm, tts, gate, transport.output(), assistant_aggregator]  # flush is instant
        [llm, gate, tts, transport.output(), assistant_aggregator]  # discard is cheaper

    Both positions keep discarded responses out of the LLM context, since the
    assistant aggregator sits at the end of the pipeline.

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
    gate is waiting for, so holding them would deadlock it.
    """

    def __init__(self, *, max_buffer_duration: float = 5.0, **kwargs):
        """Initialize the speculative response gate.

        Args:
            max_buffer_duration: Seconds to hold a speculative response before
                giving up on it and discarding it. Without this bound, a service
                that stops sending turn signals mid-speculation would leave the
                bot silent for the rest of the session.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._max_buffer_duration = max_buffer_duration
        self._state = SpeculationState.OPEN
        self._speculation_id: str | None = None
        self._buffer: list[tuple[Frame, FrameDirection]] = []
        # A speculation confirmed before its response arrived. The confirmation
        # travels as a system frame, so it can pass the queued frames it
        # confirms, and nothing follows it to correct a response held by
        # mistake — the turn is over, so no further response is coming.
        # One slot is enough: only one speculation is ever in flight.
        self._confirmed_id: str | None = None
        self._timeout_task: asyncio.Task | None = None

    @property
    def state(self) -> SpeculationState:
        """What the gate is currently doing with frames passing through it."""
        return self._state

    async def cleanup(self):
        """Clean up the gate."""
        await super().cleanup()
        await self._cancel_timeout()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Hold, release or discard a speculative response.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, SystemFrame):
            # Forwarded before the verdict is applied, so a released response
            # still follows the frame that ended the turn it answers.
            await self.push_frame(frame, direction)
            if isinstance(frame, EagerEndOfTurnCancelFrame):
                await self._discard(frame.speculation_id)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                await self._release(frame.speculation_id)
            elif isinstance(frame, InterruptionFrame):
                await self._discard(None)
            return

        if isinstance(frame, EndFrame):
            # Uninterruptible and awaited by the runner: holding it hangs shutdown.
            await self._discard(None)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            await self._begin(frame.speculation_id)

        if self._state == SpeculationState.HOLDING:
            self._buffer.append((frame, direction))
        elif self._state == SpeculationState.DROPPING:
            if isinstance(frame, LLMFullResponseEndFrame):
                self._state = SpeculationState.OPEN
        else:
            await self.push_frame(frame, direction)

    async def _begin(self, speculation_id: str | None):
        """Decide what to do with the response this id opens."""
        if self._state != SpeculationState.OPEN:
            # A new response supersedes the one we were holding or dropping. A
            # withdrawn response may never send its end frame, since its
            # generation was cancelled mid-flight, and an unconfirmed one is
            # void once something else starts answering. Nothing of it can still
            # be queued behind this frame, so there is no tail left to drop.
            await self._drop_held("superseded by a new response", keep_dropping=False)

        if not speculation_id:
            return

        if speculation_id == self._confirmed_id:
            # Confirmed before it reached us; nothing left to hold back.
            self._confirmed_id = None
            return

        self._state = SpeculationState.HOLDING
        self._speculation_id = speculation_id
        await self._start_timeout()

    async def _release(self, speculation_id: str | None):
        """Release the response the turn end confirms.

        Args:
            speculation_id: The speculation the turn confirms. A turn that ends
                without naming one confirms nothing, and releases nothing.
        """
        if not speculation_id:
            return

        if self._state != SpeculationState.HOLDING or speculation_id != self._speculation_id:
            # Confirmed before its response reached us. Remember it, or the
            # response would be held on arrival and never released: the turn is
            # over, so nothing follows to supersede it.
            self._confirmed_id = speculation_id
            return

        await self._cancel_timeout()
        logger.debug(f"{self}: releasing speculative response ({len(self._buffer)} frames)")
        buffered, self._buffer = self._buffer, []
        self._state = SpeculationState.OPEN
        self._speculation_id = None
        for frame, direction in buffered:
            await self.push_frame(frame, direction)

    async def _discard(self, speculation_id: str | None):
        """Discard the response a withdrawal or an interruption voids.

        A withdrawal that arrives before the response it voids needs no memory:
        the response is held on arrival, and whatever answers the turn instead
        supersedes it — or the buffer times out if nothing does.

        Args:
            speculation_id: The speculation being withdrawn, or None to discard
                whatever is held, which is what an interruption and shutdown do.
        """
        if speculation_id and speculation_id != self._speculation_id:
            return

        await self._drop_held("withdrawn", keep_dropping=True)

    async def _drop_held(self, reason: str, *, keep_dropping: bool):
        """Drop the held response.

        Args:
            reason: Why it is being dropped, for the log line.
            keep_dropping: Whether the rest of the response may still be queued
                behind us and has to be dropped as it arrives. False when
                something already past it proves there is no tail left.
        """
        if self._state != SpeculationState.HOLDING:
            self._state = SpeculationState.OPEN
            self._speculation_id = None
            return

        await self._cancel_timeout()
        logger.debug(
            f"{self}: discarding speculative response ({len(self._buffer)} frames, {reason})"
        )
        complete = any(isinstance(f, LLMFullResponseEndFrame) for f, _ in self._buffer)
        self._buffer.clear()
        self._speculation_id = None
        self._state = (
            SpeculationState.DROPPING if keep_dropping and not complete else SpeculationState.OPEN
        )

    async def _start_timeout(self):
        await self._cancel_timeout()
        self._timeout_task = self.create_task(self._timeout_handler(), "_speculation_timeout")

    async def _cancel_timeout(self):
        if self._timeout_task:
            task, self._timeout_task = self._timeout_task, None
            await self.cancel_task(task)

    async def _timeout_handler(self):
        await asyncio.sleep(self._max_buffer_duration)
        logger.warning(
            f"{self}: speculative response unresolved after {self._max_buffer_duration}s, "
            "discarding it"
        )
        self._timeout_task = None
        await self._discard(None)
