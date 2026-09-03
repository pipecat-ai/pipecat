#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gate that holds a speculative bot response until the turn it answers is confirmed."""

import asyncio
from collections import OrderedDict
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

# How many resolved speculations to remember. A resolution can overtake the
# response frames it resolves (it travels as a system frame, they don't), so the
# verdict has to outlive the speculation itself. Two turns' worth is plenty.
_RESOLVED_HISTORY = 8


class SpeculationState(Enum):
    """What the gate is doing with the frames passing through it.

    - ``OPEN``: forwarding everything.
    - ``BUFFERING``: holding a speculative response until it is confirmed.
    - ``DROPPING``: discarding the rest of a withdrawn speculative response.
    """

    OPEN = "open"
    BUFFERING = "buffering"
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
    service stamps with a ``speculation_id``. It is released by a
    :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` naming that id (the
    turn ended, and this response answers it) and discarded by an
    :class:`~pipecat.frames.frames.EagerEndOfTurnCancelFrame`. Every signal is
    matched by id, so a turn that ends without confirming the speculation
    releases nothing, whichever order the two arrive in.

    While buffering, everything is held in arrival order except system frames,
    which are out-of-band throughout Pipecat — and which carry the two signals
    the gate is waiting for, so holding them would deadlock it.
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
        self._resolved: OrderedDict[str, SpeculationState] = OrderedDict()
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
        """Buffer, release or discard a speculative response.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, SystemFrame):
            # Forwarded before the buffer is resolved, so a released response
            # still follows the frame that ended the turn it answers.
            await self.push_frame(frame, direction)
            if isinstance(frame, EagerEndOfTurnCancelFrame):
                await self._discard(frame.speculation_id)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                await self._release(frame.speculation_id)
            elif isinstance(frame, InterruptionFrame):
                await self._discard(self._speculation_id)
            return

        if isinstance(frame, EndFrame):
            # Uninterruptible and awaited by the runner: holding it hangs shutdown.
            await self._discard(self._speculation_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            await self._start(frame)
        elif (
            isinstance(frame, LLMFullResponseEndFrame) and self._state == SpeculationState.DROPPING
        ):
            self._state = SpeculationState.OPEN
            self._speculation_id = None
            return

        if self._state == SpeculationState.BUFFERING:
            self._buffer.append((frame, direction))
        elif self._state == SpeculationState.DROPPING:
            pass
        else:
            await self.push_frame(frame, direction)

    async def _start(self, frame: LLMFullResponseStartFrame):
        """Decide what to do with the response this frame opens."""
        if self._state == SpeculationState.DROPPING:
            # The withdrawn response never sent its end frame — its generation
            # was cancelled mid-flight. A new response is the boundary instead.
            self._state = SpeculationState.OPEN
            self._speculation_id = None

        if not frame.speculation_id:
            return

        verdict = self._resolved.get(frame.speculation_id)
        if verdict == SpeculationState.DROPPING:
            # Withdrawn before its first frame arrived: the cancellation travels
            # as a system frame, so it overtakes the response it cancels.
            self._state = SpeculationState.DROPPING
            self._speculation_id = frame.speculation_id
            return
        if verdict == SpeculationState.BUFFERING:
            # Already confirmed; nothing left to hold back.
            return

        self._state = SpeculationState.BUFFERING
        self._speculation_id = frame.speculation_id
        await self._start_timeout()

    async def _release(self, speculation_id: str | None):
        """Release the buffered response and pass the rest of it through.

        Args:
            speculation_id: The speculation the turn end confirms. A turn that
                ends without naming one confirms nothing: the response was
                generated for a prediction that missed, and the withdrawal for
                it may not have reached us yet.
        """
        if speculation_id and speculation_id != self._speculation_id:
            # Confirmed before the response reached us. Record the verdict so it
            # passes straight through on arrival.
            self._remember(speculation_id, SpeculationState.BUFFERING)
            return

        if self._state != SpeculationState.BUFFERING or not speculation_id:
            return

        logger.debug(f"{self}: releasing speculative response ({len(self._buffer)} frames)")
        await self._resolve(SpeculationState.BUFFERING)
        buffered, self._buffer = self._buffer, []
        for frame, direction in buffered:
            await self.push_frame(frame, direction)

    async def _discard(self, speculation_id: str | None):
        """Discard the buffered response and the rest of the one it belongs to.

        Args:
            speculation_id: The speculation to withdraw, or None to withdraw
                whatever is buffered — which is what an interruption does.
        """
        if speculation_id and speculation_id != self._speculation_id:
            # Not the response we're holding — it hasn't reached us yet, since
            # the withdrawal travels as a system frame and overtakes it. Record
            # the verdict so it is discarded on arrival.
            self._remember(speculation_id, SpeculationState.DROPPING)
            return

        if self._state != SpeculationState.BUFFERING:
            return

        logger.debug(f"{self}: discarding speculative response ({len(self._buffer)} frames)")
        complete = any(isinstance(f, LLMFullResponseEndFrame) for f, _ in self._buffer)
        self._buffer.clear()
        await self._resolve(SpeculationState.DROPPING)
        if not complete:
            # The rest of the response may still be queued behind us; keep
            # dropping until it ends or another one starts.
            self._state = SpeculationState.DROPPING

    async def _resolve(self, verdict: SpeculationState):
        await self._cancel_timeout()
        if self._speculation_id:
            self._remember(self._speculation_id, verdict)
        self._state = SpeculationState.OPEN

    def _remember(self, speculation_id: str, verdict: SpeculationState):
        self._resolved[speculation_id] = verdict
        self._resolved.move_to_end(speculation_id)
        while len(self._resolved) > _RESOLVED_HISTORY:
            self._resolved.popitem(last=False)

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
        await self._discard(self._speculation_id)
