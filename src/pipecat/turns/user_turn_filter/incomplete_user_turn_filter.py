#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn filter that ends a turn only when the LLM marks it complete."""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    UserTurnCompletionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_filter.base_user_turn_filter import BaseUserTurnFilter


class IncompleteUserTurnFilter(BaseUserTurnFilter):
    """Ends a user turn only when the LLM marks it complete.

    Turns on turn-completion markers in the LLM, which then starts every
    response with ● (the turn is complete), ◐ (the user will continue shortly)
    or ○ (the user needs more time). The LLM reports each marker as a
    :class:`~pipecat.frames.frames.UserTurnCompletionFrame`.

    ● accepts the proposed turn end, and so does a response with no marker. ◐
    and ○ reject it: the answer is dropped and the turn stays open. If the user
    then stays quiet for the configured timeout, the filter ends the turn and
    asks the LLM to re-engage them, even when the controller's turn stop
    timeout ended the turn first. A ● that arrives while the user is speaking
    again is stale, and is taken as ◐.

    Example::

        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(user_turn_filter=IncompleteUserTurnFilter()),
        )
    """

    def __init__(
        self,
        *,
        config: UserTurnCompletionConfig | None = None,
        **kwargs,
    ):
        """Initialize the incomplete user turn filter.

        Args:
            config: Markers, instructions, timeouts and re-prompts. Defaults to
                ``UserTurnCompletionConfig()``.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self._config = config or UserTurnCompletionConfig()
        # The inference whose marker decides the proposal being judged.
        self._awaited_id: int | None = None
        # The latest report per inference, kept for a proposal it arrives ahead of.
        self._reports: dict[int | None, UserTurnCompletionFrame] = {}
        self._user_speaking = False
        self._reprompt_task: asyncio.Task | None = None

    @property
    def config(self) -> UserTurnCompletionConfig:
        """The turn completion configuration applied to the LLM."""
        return self._config

    async def cleanup(self):
        """Cleanup the filter."""
        await super().cleanup()
        await self._cancel_reprompt()

    async def handle_user_turn_started(self):
        """Forget the previous turn's reports and cancel any pending re-prompt."""
        self._forget_reports()
        await self._cancel_reprompt()

    async def handle_user_turn_stopped(self):
        """Forget this turn's reports, keeping a pending re-prompt."""
        self._forget_reports()

    async def handle_turn_end_proposed(self, context_frame_id: int):
        """Judge the proposal by the marker of the inference generated for it.

        Args:
            context_frame_id: The id of the ``LLMContextFrame`` that started it.
        """
        self._awaited_id = context_frame_id
        report = self._reports.get(context_frame_id)
        if report:
            await self._judge(report)

    async def process_frame(self, frame: Frame):
        """Configure the LLM on start and follow its reports and the user's speech.

        Args:
            frame: The frame to be analyzed.
        """
        if isinstance(frame, StartFrame):
            await self._configure_llm()
        elif isinstance(frame, UserTurnCompletionFrame):
            await self._handle_report(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            # The user is talking again, so they don't need re-engaging.
            await self._cancel_reprompt()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False

    async def _configure_llm(self):
        """Turn on the markers in every LLM, including inactive ones behind a switcher."""
        await self.push_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    filter_incomplete_user_turns=True,
                    user_turn_completion_config=self._config,
                ),
                reach_inactive_services=True,
            )
        )

    async def _handle_report(self, report: UserTurnCompletionFrame):
        self._reports[report.context_frame_id] = report
        if self._awaited_id is not None and report.context_frame_id == self._awaited_id:
            await self._judge(report)

    async def _judge(self, report: UserTurnCompletionFrame):
        """Answer the proposal from the LLM's marker."""
        self._awaited_id = None
        if report.kind in ("short", "long"):
            await self._reject(report.kind)
        elif report.kind == "complete" and self._user_speaking:
            logger.debug(f"{self}: turn marked complete while the user speaks, taking it as short")
            await self._reject("short")
        else:
            await self._cancel_reprompt()
            await self.accept()

    async def _reject(self, kind: str):
        await self.reject()
        await self._start_reprompt(kind)

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
        await self.accept()
        await self.push_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "developer", "content": prompt}], run_llm=True
            )
        )

    async def _cancel_reprompt(self):
        if self._reprompt_task:
            task, self._reprompt_task = self._reprompt_task, None
            await self.cancel_task(task)

    def _forget_reports(self):
        self._awaited_id = None
        self._reports.clear()
