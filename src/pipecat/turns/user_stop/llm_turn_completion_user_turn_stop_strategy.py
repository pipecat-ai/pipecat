#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy gated on the LLM's turn-completion verdict."""

from pipecat.frames.frames import (
    Frame,
    LLMUpdateSettingsFrame,
    StartFrame,
    UserTurnCompletedFrame,
)
from pipecat.services.settings import LLMSettings
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig


class LLMTurnCompletionUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy that finalizes only when the LLM agrees.

    This strategy lets another stop strategy (e.g. smart-turn analyzer)
    trigger LLM inference, then defers the public ``on_user_turn_stopped``
    event until the LLM emits a ``UserTurnCompletedFrame``. On
    ``incomplete_short`` / ``incomplete_long`` markers the
    :class:`~pipecat.turns.user_turn_completion_mixin.UserTurnCompletionLLMServiceMixin`
    re-prompts the LLM internally and no completion frame is emitted.

    To use this strategy, install it alongside one or more upstream stop
    strategies in ``UserTurnStrategies.stop`` and wrap those upstream
    strategies with :func:`~pipecat.turns.user_stop.deferred` so they
    fire only ``on_user_turn_inference_triggered`` and leave
    finalization to this strategy. The aggregator's deprecation path
    for ``filter_incomplete_user_turns`` does this rewiring
    automatically.

    If the LLM never returns a completion frame (malformed output,
    unreachable service, etc.), the controller's
    ``user_turn_stop_timeout`` watchdog is the safety net — it fires
    ``on_user_turn_stopped`` after no activity for that many seconds.
    Tune ``user_turn_stop_timeout`` higher if your LLM regularly takes
    longer than the default to respond.

    On ``StartFrame`` the strategy pushes an ``LLMUpdateSettingsFrame``
    upstream that enables ``filter_incomplete_user_turns`` on the LLM
    service and seeds the
    :class:`~pipecat.turns.user_turn_completion_mixin.UserTurnCompletionConfig`.
    """

    def __init__(
        self,
        *,
        config: UserTurnCompletionConfig | None = None,
        **kwargs,
    ):
        """Initialize the LLM turn-completion stop strategy.

        Args:
            config: Configuration applied to the LLM via the
                ``filter_incomplete_user_turns`` setting on
                ``StartFrame``. Defaults to ``UserTurnCompletionConfig()``.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """
        super().__init__(**kwargs)
        self._config = config or UserTurnCompletionConfig()

    @property
    def config(self) -> UserTurnCompletionConfig:
        """Return the configured ``UserTurnCompletionConfig``."""
        return self._config

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Observe frames to drive the finalization decision."""
        if isinstance(frame, StartFrame):
            await self._configure_llm()
        elif isinstance(frame, UserTurnCompletedFrame):
            await self.trigger_user_turn_finalized()

        return ProcessFrameResult.CONTINUE

    async def _configure_llm(self):
        await self.push_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    filter_incomplete_user_turns=True,
                    user_turn_completion_config=self._config,
                )
            )
        )
