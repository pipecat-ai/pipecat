#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn start strategy configuration."""

from dataclasses import dataclass

from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    ExternalUserTurnStopStrategy,
    LLMTurnCompletionUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
    deferred,
)
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig


def default_user_turn_start_strategies() -> list[BaseUserTurnStartStrategy]:
    """Return the default user turn start strategies.

    Returns ``[VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]``.
    Useful when building a custom strategy list that extends the defaults.

    Example::

        start_strategies = [
            WakePhraseUserTurnStartStrategy(phrases=["hey pipecat"]),
            *default_user_turn_start_strategies(),
        ]
    """
    return [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]


def default_user_turn_stop_strategies() -> list[BaseUserTurnStopStrategy]:
    """Return the default user turn stop strategies.

    Returns ``[TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)]``.
    Useful when building a custom strategy list that extends the defaults.
    """
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    return [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]


def llm_completion_user_turn_stop_strategies(
    strategies: list[BaseUserTurnStopStrategy] | None = None,
    *,
    config: UserTurnCompletionConfig | None = None,
) -> list[BaseUserTurnStopStrategy]:
    """Build a stop-strategy list gated on the LLM's turn-completion verdict.

    Wraps ``strategies`` with :func:`deferred` so they trigger inference
    but don't fire ``on_user_turn_stopped`` themselves, then appends
    :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`
    as the finalizer. Use as the ``stop`` field of a
    :class:`UserTurnStrategies`::

        UserTurnStrategies(
            stop=llm_completion_user_turn_stop_strategies(),
        )

    Args:
        strategies: Stop strategies that should drive inference. If
            None, uses :func:`default_user_turn_stop_strategies`.
        config: Optional configuration applied to the LLM via the
            ``filter_incomplete_user_turns`` setting. Customizes the
            turn-completion instructions, incomplete-turn timeouts, and
            re-prompts.

    Returns:
        ``[deferred(s) for s in strategies] +
        [LLMTurnCompletionUserTurnStopStrategy(config=config)]``.
    """
    strategies = strategies if strategies is not None else default_user_turn_stop_strategies()
    return [
        *(deferred(s) for s in strategies),
        LLMTurnCompletionUserTurnStopStrategy(config=config),
    ]


@dataclass
class UserTurnStrategies:
    """Container for user turn start and stop strategies.

    If no strategies are specified, the following defaults are used:

        start: [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]
         stop: [TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)]

    Parameters:
        start: A list of user turn start strategies used to detect when
            the user starts speaking.
        stop: A list of user turn stop strategies used to decide when
            the user stops speaking.

    """

    start: list[BaseUserTurnStartStrategy] | None = None
    stop: list[BaseUserTurnStopStrategy] | None = None

    def __post_init__(self):
        if not self.start:
            self.start = default_user_turn_start_strategies()
        if not self.stop:
            self.stop = default_user_turn_stop_strategies()


@dataclass
class ExternalUserTurnStrategies(UserTurnStrategies):
    """Default container for external user turn start and stop strategies.

    This class provides a convenience default for configuring external turn
    control. It preconfigures `UserTurnStrategies` with
    `ExternalUserTurnStartStrategy` and `ExternalUserTurnStopStrategy`, allowing
    external processors (such as services) to control when user turn starts and
    stops.

    When using this container, the user aggregator does not push
    `UserStartedSpeakingFrame` or `UserStoppedSpeakingFrame` frames, and does
    not generate interruptions. These signals are expected to be provided by an
    external processor.

    """

    def __post_init__(self):
        self.start = [ExternalUserTurnStartStrategy()]
        self.stop = [ExternalUserTurnStopStrategy()]
