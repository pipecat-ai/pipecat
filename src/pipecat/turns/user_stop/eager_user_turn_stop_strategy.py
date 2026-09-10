#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that answers an eager end of turn speculatively."""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerTranscriptionFrame,
    Frame,
)
from pipecat.turns.types import ProcessFrameResult, UserTurnSpeculation
from pipecat.turns.user_stop.eager_match_policy import EagerMatchPolicy, NormalizedMatch
from pipecat.turns.user_stop.external_user_turn_stop_strategy import ExternalUserTurnStopStrategy


class EagerUserTurnStopStrategy(ExternalUserTurnStopStrategy):
    """Answers an eager end of turn while the turn is still open.

    Some STT services predict the end of a turn before committing to it, and
    withdraw the prediction if the user turns out to be mid-sentence. This
    strategy starts generating a response on that prediction, so the gap between
    it and the committed end of turn is spent generating rather than waiting.

    The prediction can be wrong in three ways, and all three discard the
    response with an
    :class:`~pipecat.frames.frames.EagerEndOfTurnCancelFrame`:

    - the user resumes speaking, and the service withdraws the eager end of turn
    - the committed transcript differs from the eager one, per ``match_policy``;
      the turn then ends without confirming the speculation, and the aggregator
      withdraws it ahead of the turn end
    - the service never commits or withdraws within ``speculation_timeout``

    Nothing the speculation produces reaches the user or the context. The
    inference runs against a provisional context, and its response is held by
    the LLM service's
    :class:`~pipecat.turns.speculation_gate.SpeculationGate` until the turn
    ends. The turn ends normally: the user message written to the context is
    always the committed transcript, never the eager one.

    Install it with :class:`~pipecat.turns.user_turn_strategies.EagerUserTurnStrategies`
    rather than directly — the service owns turn detection here, so it replaces
    the detector chain instead of running alongside it.
    """

    def __init__(
        self,
        *,
        match_policy: EagerMatchPolicy | None = None,
        speculation_timeout: float = 5.0,
        **kwargs,
    ):
        """Initialize the eager user turn stop strategy.

        Args:
            match_policy: Decides whether the committed transcript is close
                enough to the eager one to keep the speculative response.
                Defaults to :class:`~pipecat.turns.user_stop.NormalizedMatch`,
                which ignores the capitalization and punctuation services
                commonly add when they commit a transcript. Pass
                :class:`~pipecat.turns.user_stop.ExactMatch` to require the two
                to be identical.
            speculation_timeout: Seconds a speculation may wait for the service
                to commit or withdraw the eager end of turn before the strategy
                withdraws it. Without this bound, a service that stops sending
                turn signals mid-speculation would leave the response held, and
                the bot silent, for the rest of the session.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self._match_policy = match_policy or NormalizedMatch()
        self._speculation_timeout = speculation_timeout
        self._speculation: UserTurnSpeculation | None = None
        self._timeout_task: asyncio.Task | None = None

    @property
    def match_policy(self) -> EagerMatchPolicy:
        """The policy deciding whether a speculative response still applies."""
        return self._match_policy

    async def cleanup(self):
        """Stop bounding a speculation that outlived the pipeline."""
        await self._settle()
        await super().cleanup()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Start a speculation on an eager end of turn, or withdraw one.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always CONTINUE, so subsequent stop strategies are evaluated.
        """
        if isinstance(frame, EagerTranscriptionFrame):
            await self._speculate(frame)
        elif isinstance(frame, EagerEndOfTurnCancelFrame):
            # The service withdrew its prediction, and its frame reaches every
            # consumer on its own. Only our own state is left to clear.
            await self._settle()

        return await super().process_frame(frame)

    async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames: bool | None = None):
        """End the turn, keeping the speculative response only if it still applies.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn.
        """
        speculation = await self._settle()
        if not speculation:
            await super().trigger_user_turn_stopped(
                enable_user_speaking_frames=enable_user_speaking_frames
            )
            return

        if self._match_policy.matches(speculation.text, self._text):
            logger.debug(f"{self}: eager end of turn held, keeping the speculative response")
            # Inference already ran, on the eager transcript. Only finalize: the
            # UserStoppedSpeakingFrame that ends the turn releases the response.
            await self.trigger_user_turn_finalized(
                enable_user_speaking_frames=enable_user_speaking_frames, speculated=True
            )
            return

        logger.debug(
            f"{self}: eager end of turn missed, discarding the speculative response "
            f"(eager: [{speculation.text}], committed: [{self._text}])"
        )
        # Only finalize, without confirming the speculation: the aggregator
        # withdraws it ahead of the turn end and runs inference again, on the
        # committed transcript, behind it. Withdrawing from here would queue
        # the cancel behind frames the aggregator is processing, and the turn
        # end would overtake it and release the response.
        await self.trigger_user_turn_finalized(
            enable_user_speaking_frames=enable_user_speaking_frames
        )

    async def handle_user_turn_started(self):
        """Start a turn, withdrawing a speculation the previous turn left in flight."""
        if await self._settle():
            # The previous turn never ended, so nothing withdrew its
            # speculation, and the response would be held for good. A turn that
            # ends without confirming its speculation is withdrawn by the
            # aggregator, ahead of the turn end.
            logger.debug(f"{self}: new turn over a live speculation, discarding its response")
            await self.push_frame(EagerEndOfTurnCancelFrame())
        await super().handle_user_turn_started()

    async def _reset(self):
        """Clear per-turn state. Runs at both turn boundaries."""
        await self._settle()
        await super()._reset()

    async def _speculate(self, frame: EagerTranscriptionFrame):
        """Answer an eager end of turn, leaving the turn open."""
        # A prediction the service replaces without withdrawing: the response
        # to the old one answers nothing, so it is withdrawn here.
        if await self._settle():
            await self.push_frame(EagerEndOfTurnCancelFrame())
        # Segments committed earlier in this turn are part of what the LLM will
        # see, so they're part of what the committed transcript is compared to.
        self._speculation = UserTurnSpeculation(text=self._text + frame.text)
        self._timeout_task = self.create_task(
            self._speculation_timeout_handler(), f"{self}::_speculation_timeout"
        )
        logger.debug(f"{self}: speculating on eager end of turn: [{self._speculation.text}]")
        await self.trigger_user_turn_inference_triggered(speculation=self._speculation)

    async def _settle(self) -> UserTurnSpeculation | None:
        """Take the speculation in flight, if any, and stop bounding it."""
        speculation, self._speculation = self._speculation, None
        task, self._timeout_task = self._timeout_task, None
        if task and task is not asyncio.current_task():
            await self.cancel_task(task)
        return speculation

    async def _speculation_timeout_handler(self):
        """Withdraw a speculation the service neither committed nor withdrew in time."""
        await asyncio.sleep(self._speculation_timeout)
        if not await self._settle():
            return
        logger.warning(
            f"{self}: eager end of turn unresolved after {self._speculation_timeout}s, "
            "discarding the speculative response"
        )
        await self.push_frame(EagerEndOfTurnCancelFrame())
