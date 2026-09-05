#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that answers an eager end of turn speculatively."""

from loguru import logger

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerEndOfTurnTranscriptionFrame,
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

    The prediction can be wrong in two ways, and both discard the response:

    - the user resumes speaking, and the service withdraws the eager end of turn
    - the committed transcript differs from the eager one, per ``match_policy``

    Nothing the speculation produces reaches the user or the context. The
    inference runs against a provisional context, and its response is held by a
    :class:`~pipecat.processors.filters.user_turn_speculation_gate.UserTurnSpeculationGate`,
    which must be in the pipeline before the output transport. The turn ends
    normally: the user message written to the context is always the committed
    transcript, never the eager one.

    Install it with :class:`~pipecat.turns.user_turn_strategies.EagerUserTurnStrategies`
    rather than directly — the service owns turn detection here, so it replaces
    the detector chain instead of running alongside it.
    """

    def __init__(self, *, match_policy: EagerMatchPolicy | None = None, **kwargs):
        """Initialize the eager user turn stop strategy.

        Args:
            match_policy: Decides whether the committed transcript is close
                enough to the eager one to keep the speculative response.
                Defaults to :class:`~pipecat.turns.user_stop.NormalizedMatch`,
                which ignores the capitalization and punctuation services
                commonly add when they commit a transcript. Pass
                :class:`~pipecat.turns.user_stop.ExactMatch` to require the two
                to be identical.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self._match_policy = match_policy or NormalizedMatch()
        self._speculation: UserTurnSpeculation | None = None

    @property
    def match_policy(self) -> EagerMatchPolicy:
        """The policy deciding whether a speculative response still applies."""
        return self._match_policy

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Start a speculation on an eager end of turn, or withdraw one.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always CONTINUE, so subsequent stop strategies are evaluated.
        """
        if isinstance(frame, EagerEndOfTurnTranscriptionFrame):
            await self._speculate(frame)
        elif isinstance(frame, EagerEndOfTurnCancelFrame):
            # The service withdrew its prediction, and its frame reaches every
            # consumer on its own. Only our own state is left to clear.
            self._forget(frame.speculation_id)

        return await super().process_frame(frame)

    async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames: bool | None = None):
        """End the turn, keeping the speculative response only if it still applies.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn.
        """
        speculation = self._speculation
        if not speculation:
            await super().trigger_user_turn_stopped(
                enable_user_speaking_frames=enable_user_speaking_frames
            )
            return

        self._speculation = None

        if self._match_policy.matches(speculation.text, self._text):
            logger.debug(f"{self}: eager end of turn held, keeping the speculative response")
            # Inference already ran, on the eager transcript. Only finalize,
            # naming the speculation: the UserStoppedSpeakingFrame that carries
            # its id is what releases the response.
            await self.trigger_user_turn_finalized(
                enable_user_speaking_frames=enable_user_speaking_frames,
                speculation_id=speculation.id,
            )
            return

        logger.debug(
            f"{self}: eager end of turn missed, discarding the speculative response "
            f"(eager: [{speculation.text}], committed: [{self._text}])"
        )
        await self.push_frame(EagerEndOfTurnCancelFrame(speculation.id))
        # Inference has to run again, on the committed transcript, so fire both
        # events rather than just finalizing.
        await super().trigger_user_turn_stopped(
            enable_user_speaking_frames=enable_user_speaking_frames
        )

    async def _reset(self):
        """Clear per-turn state. Runs at both turn boundaries."""
        speculation = self._speculation
        await super()._reset()
        self._speculation = None
        if speculation:
            # The turn ended without resolving the speculation — the stop
            # watchdog, an interruption, session end. Nothing else will withdraw
            # it, so the response would be held until the gate times out.
            logger.debug(f"{self}: turn ended unresolved, discarding the speculative response")
            await self.push_frame(EagerEndOfTurnCancelFrame(speculation.id))

    async def _speculate(self, frame: EagerEndOfTurnTranscriptionFrame):
        """Answer an eager end of turn, leaving the turn open."""
        # Segments committed earlier in this turn are part of what the LLM will
        # see, so they're part of what the committed transcript is compared to.
        self._speculation = UserTurnSpeculation(
            id=frame.speculation_id, text=self._text + frame.text
        )
        logger.debug(f"{self}: speculating on eager end of turn: [{self._speculation.text}]")
        await self.trigger_user_turn_inference_triggered(speculation=self._speculation)

    def _forget(self, speculation_id: str):
        """Drop a speculation the service withdrew."""
        if self._speculation and self._speculation.id == speculation_id:
            self._speculation = None
