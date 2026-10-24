#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy gated on the LLM's turn-completion verdict."""

import warnings

from pipecat.frames.frames import Frame, LLMUpdateSettingsFrame, StartFrame
from pipecat.services.settings import LLMSettings
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.external_user_turn_completion_stop_strategy import (
    ExternalUserTurnCompletionStopStrategy,
)
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.utils.deprecation import deprecated

# Subclasses the deprecated ExternalUserTurnCompletionStopStrategy; silence the
# subclassing DeprecationWarning that @deprecated would otherwise emit while
# this module is imported.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

    @deprecated(
        "`LLMTurnCompletionUserTurnStopStrategy` is deprecated since 1.14.0 and will be removed "
        "in 2.0.0. Use `IncompleteUserTurnFilter` instead."
    )
    class LLMTurnCompletionUserTurnStopStrategy(ExternalUserTurnCompletionStopStrategy):
        """LLM-gated stop strategy.

        .. deprecated:: 1.14.0
            Use :class:`~pipecat.turns.user_turn_filter.IncompleteUserTurnFilter` instead.
            Will be removed in 2.0.0.

        On ``StartFrame``, pushes an ``LLMUpdateSettingsFrame`` downstream that
        enables ``filter_incomplete_user_turns`` on the LLM and seeds the
        :class:`~pipecat.turns.user_turn_completion_mixin.UserTurnCompletionConfig`.
        The update is marked ``reach_inactive_services`` so that every LLM behind
        an :class:`~pipecat.pipeline.llm_switcher.LLMSwitcher` is configured, not
        only the one active at startup.

        The user aggregator turns a setup that uses this strategy into an
        :class:`~pipecat.turns.user_turn_filter.IncompleteUserTurnFilter` over its
        ``deferred(...)`` detectors, so the strategy itself doesn't run there.
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
                    ),
                    reach_inactive_services=True,
                )
            )
