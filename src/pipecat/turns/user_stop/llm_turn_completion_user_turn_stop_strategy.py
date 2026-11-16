#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that decides on the LLM's turn-completion marker."""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMUpdateSettingsFrame,
    ProposedUserStartedSpeakingFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserTurnCompletionFrame,
    UserTurnResponseCancelFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig


class LLMTurnCompletionUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Decides the user is done only when the LLM marks the turn complete.

    Turns on turn-completion markers in the LLM, which then starts every
    response with ● (the turn is complete), ◐ (the user will continue shortly)
    or ○ (the user needs more time), and reports each marker as a
    :class:`~pipecat.frames.frames.UserTurnCompletionFrame`. The settings update
    reaches every LLM behind an
    :class:`~pipecat.pipeline.llm_switcher.LLMSwitcher`, not only the active one.

    The strategy waits for the turn's inference rather than listening to the
    user, so it needs other stop strategies to decide the user stopped: once
    they have decided, the inference runs, as
    :class:`~pipecat.turns.user_turn_strategies.FilterIncompleteUserTurnStrategies`
    sets up. It decides the user is done on that inference's ●, on a tool
    call, or on a response with no marker. ◐ and ○ leave the turn open. If
    the user then stays quiet for the configured timeout, the strategy
    decides the user is done and asks the LLM to re-engage them, and the
    controller's turn stop timeout waits for that decision. Another
    inference answering the user in the meantime, such as one for words
    transcribed after the turn's inference, cancels the re-prompt. When the
    user speaks again, it waits for the next inference instead.

    An eager end of turn answers the turn speculatively, and the strategy
    decides on that answer too. ◐ or ○ withdraws it, and the committed turn
    brings an inference of its own.
    """

    def __init__(
        self,
        *,
        config: UserTurnCompletionConfig | None = None,
        **kwargs,
    ):
        """Initialize the LLM turn-completion stop strategy.

        Args:
            config: Markers, instructions, timeouts and re-prompts applied to the
                LLM. Defaults to ``UserTurnCompletionConfig()``.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """
        super().__init__(**kwargs)
        self._config = config or UserTurnCompletionConfig()
        # The inference whose marker this strategy decides on.
        self._awaited: LLMContextFrame | None = None
        self._reprompt_task: asyncio.Task | None = None

    @property
    def config(self) -> UserTurnCompletionConfig:
        """Return the configured ``UserTurnCompletionConfig``."""
        return self._config

    @property
    def decision_scheduled(self) -> bool:
        """Whether a re-prompt is scheduled, which decides when it fires."""
        return self._reprompt_task is not None

    @property
    def waits_for_inference(self) -> bool:
        """Decides on the LLM's marker, which only the turn's inference produces."""
        return True

    async def cleanup(self):
        """Cancel any pending re-prompt."""
        await super().cleanup()
        await self._cancel_reprompt()

    async def handle_user_turn_started(self):
        """Forget the previous turn's inference and cancel any pending re-prompt."""
        self._awaited = None
        await self._cancel_reprompt()

    async def handle_user_turn_stopped(self):
        """Forget this turn's inference, keeping a pending re-prompt."""
        self._awaited = None

    async def process_frame(self, frame: Frame):
        """Configure the LLM on start, and follow the turn's inference and the user.

        Args:
            frame: The frame to be analyzed.
        """
        if isinstance(frame, StartFrame):
            await self._configure_llm()
        elif isinstance(frame, LLMContextFrame):
            # An inference answering the user makes a pending re-prompt redundant.
            self._awaited = frame
            await self._cancel_reprompt()
        elif isinstance(frame, UserTurnCompletionFrame):
            await self._handle_report(frame)
        elif isinstance(frame, UserTurnResponseCancelFrame):
            # The answer it was waiting for is gone.
            self._awaited = None
        elif isinstance(
            frame,
            (
                VADUserStartedSpeakingFrame,
                UserStartedSpeakingFrame,
                ProposedUserStartedSpeakingFrame,
            ),
        ):
            await self._handle_user_started_speaking()

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

    async def _handle_user_started_speaking(self):
        # The user speaking again clears the decisions, and the next inference
        # brings its own marker. The user doesn't need re-engaging either.
        self._awaited = None
        await self._cancel_reprompt()

    async def _handle_report(self, report: UserTurnCompletionFrame):
        if self._awaited is None or report.context_frame_id != self._awaited.id:
            # Another inference is answering the user, so a pending re-prompt
            # would answer them twice.
            await self._cancel_reprompt()
            return

        speculation = self._awaited.speculation
        self._awaited = None
        if report.kind in ("short", "long") and speculation:
            await self.cancel_user_turn_speculation()
        elif report.kind in ("short", "long"):
            await self._start_reprompt(report.kind)
        else:
            await self._cancel_reprompt()
            await self.report_user_turn_stopped()

    async def _start_reprompt(self, kind: str):
        await self._cancel_reprompt()
        if kind == "short":
            timeout, prompt = self._config.incomplete_short_timeout, self._config.short_prompt
        else:
            timeout, prompt = self._config.incomplete_long_timeout, self._config.long_prompt
        self._reprompt_task = self.create_task(
            self._reprompt_handler(timeout, prompt), f"{self}::_reprompt_handler"
        )

    async def _reprompt_handler(self, timeout: float, prompt: str):
        """End the turn the user went quiet in, and ask the LLM to re-engage them."""
        await asyncio.sleep(timeout)
        self._reprompt_task = None
        logger.debug(f"{self}: user quiet for {timeout}s after an incomplete turn, re-prompting")
        await self.report_user_turn_stopped()
        # The turn end runs an inference of its own for words transcribed after
        # the turn's inference, and that inference answers the user instead.
        if not self._awaited:
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "developer", "content": prompt}], run_llm=True
                )
            )

    async def _cancel_reprompt(self):
        if self._reprompt_task:
            task, self._reprompt_task = self._reprompt_task, None
            await self.cancel_task(task)
