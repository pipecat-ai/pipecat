#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy gated on the LLM's turn-completion verdict."""

from pipecat.frames.frames import Frame, LLMUpdateSettingsFrame, StartFrame
from pipecat.services.settings import LLMSettings
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.external_user_turn_completion_stop_strategy import (
    ExternalUserTurnCompletionStopStrategy,
)
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig


class LLMTurnCompletionUserTurnStopStrategy(ExternalUserTurnCompletionStopStrategy):
    """LLM-gated stop strategy.

    Extends
    :class:`~pipecat.turns.user_stop.ExternalUserTurnCompletionStopStrategy`
    with the LLM-specific setup needed for the marker-based completion
    protocol: on ``StartFrame``, pushes an ``LLMUpdateSettingsFrame``
    upstream that enables ``filter_incomplete_user_turns`` on the LLM
    and seeds the
    :class:`~pipecat.turns.user_turn_completion_mixin.UserTurnCompletionConfig`.

    Finalization itself is inherited: when the LLM service's
    :class:`~pipecat.turns.user_turn_completion_mixin.UserTurnCompletionLLMServiceMixin`
    detects a ``✓`` marker, it broadcasts a
    :class:`~pipecat.frames.frames.UserTurnCompletedFrame` and the
    base class fires ``on_user_turn_stopped``. On
    ``incomplete_short`` / ``incomplete_long`` markers the mixin
    re-prompts internally and no completion frame is emitted, so the
    public stop event stays deferred.

    Install alongside one or more ``deferred(...)``-wrapped detector
    strategies that drive ``on_user_turn_inference_triggered`` but
    leave finalization to this strategy. The aggregator's deprecation
    path for ``filter_incomplete_user_turns`` does this rewiring
    automatically.
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
        """Configure the LLM on start and delegate completion handling to the base."""
        if isinstance(frame, StartFrame):
            await self._configure_llm()
        return await super().process_frame(frame)

    async def _configure_llm(self):
        await self.push_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    filter_incomplete_user_turns=True,
                    user_turn_completion_config=self._config,
                )
            )
        )
