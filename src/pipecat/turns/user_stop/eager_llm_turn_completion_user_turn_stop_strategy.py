#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that answers an eager end of turn and waits for the LLM's verdict."""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMUpdateSettingsFrame,
    ProposedUserStartedSpeakingFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.services.settings import LLMSettings
from pipecat.turns.types import ProcessFrameResult, UserTurnSpeculation
from pipecat.turns.user_stop.eager_user_turn_stop_strategy import EagerUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig


class EagerLLMTurnCompletionUserTurnStopStrategy(EagerUserTurnStopStrategy):
    """Answers an eager end of turn, and ends the turn once the LLM says it's complete.

    Combines :class:`~pipecat.turns.user_stop.EagerUserTurnStopStrategy` with the
    LLM's turn-completion markers (see
    :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`). The
    response generated on the service's eager end of turn starts with the marker,
    so the LLM's verdict comes with the response instead of after it.

    When the service commits to the end of the turn, the turn ends once that
    response's marker is ● (complete): right away if it already arrived, or when
    it does. A turn the service ends with no eager end of turn, or with a
    different transcript, is answered speculatively the same way.

    A ◐ or ○ (incomplete) keeps the turn open and the response unspoken:

    - if the user goes on, the response is withdrawn, and the next end of turn
      is answered with everything they said
    - if they don't, the controller's ``user_turn_stop_timeout`` ends the turn
      and the LLM answers it as usual

    Install it with
    :class:`~pipecat.turns.user_turn_strategies.FilterIncompleteEagerUserTurnStrategies`.
    """

    def __init__(self, *, config: UserTurnCompletionConfig | None = None, **kwargs):
        """Initialize the strategy.

        Args:
            config: Configuration applied to the LLM via the
                ``filter_incomplete_user_turns`` setting on ``StartFrame``.
                Defaults to ``UserTurnCompletionConfig()``.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~pipecat.turns.user_stop.EagerUserTurnStopStrategy`.
        """
        super().__init__(**kwargs)
        self._config = config or UserTurnCompletionConfig()
        # The speculation the LLM said completes the turn.
        self._completed: UserTurnSpeculation | None = None
        # Whether the service ended the turn, which then ends once the LLM says
        # it's complete, and whether to emit the turn frames when it does.
        self._ending = False
        self._ending_user_speaking_frames: bool | None = None

    @property
    def config(self) -> UserTurnCompletionConfig:
        """Return the configured ``UserTurnCompletionConfig``."""
        return self._config

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Configure the LLM, and follow its verdict and the user going on.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always CONTINUE, so subsequent stop strategies are evaluated.
        """
        if isinstance(frame, StartFrame):
            await self._configure_llm()

        result = await super().process_frame(frame)

        if isinstance(frame, UserTurnInferenceCompletedFrame):
            await self._handle_turn_complete()
        elif isinstance(frame, (ProposedUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            await self._handle_user_went_on()

        return result

    async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames: bool | None = None):
        """End the turn once the LLM says it's complete, answering it if nothing does yet.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn.
        """
        # The base strategy calls this again while the turn stays open.
        if self._ending:
            return

        self._ending = True
        self._ending_user_speaking_frames = enable_user_speaking_frames

        speculation = self._speculation
        if speculation and not self._match_policy.matches(speculation.text, self._text):
            logger.debug(
                f"{self}: eager end of turn missed, discarding the speculative response "
                f"(eager: [{speculation.text}], committed: [{self._text}])"
            )
            await self._take_speculation()
            await self.trigger_user_turn_speculation_cancelled()
            speculation = None

        if speculation is None:
            await self._start_speculation(self._text)
        elif self._completed is speculation:
            await self._end_turn()
        else:
            logger.debug(f"{self}: waiting for the LLM to say the turn is complete")

    async def _reset(self):
        """Clear per-turn state. Runs at both turn boundaries."""
        self._completed = None
        self._ending = False
        await super()._reset()

    async def _configure_llm(self):
        await self.push_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    filter_incomplete_user_turns=True,
                    user_turn_completion_config=self._config,
                ),
                reach_inactive_services=True,
            )
        )

    async def _handle_turn_complete(self):
        """Note the LLM saying the turn is complete, ending it if the service has."""
        if self._speculation is None:
            return
        self._completed = self._speculation
        if self._ending:
            await self._end_turn()

    async def _handle_user_went_on(self):
        """Withdraw the response to a turn the user went on with after the service ended it."""
        if not self._ending:
            return
        self._ending = False
        if await self._take_speculation():
            logger.debug(f"{self}: the user went on, discarding the speculative response")
            await self.trigger_user_turn_speculation_cancelled()

    async def _end_turn(self):
        """End the turn, keeping the response the LLM said completes it."""
        await self._take_speculation()
        self._ending = False
        logger.debug(f"{self}: the LLM says the turn is complete, keeping the speculative response")
        await self.trigger_user_turn_finalized(
            enable_user_speaking_frames=self._ending_user_speaking_frames,
            confirms_speculation=True,
        )
