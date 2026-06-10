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


@dataclass
class FilterIncompleteUserTurnStrategies(UserTurnStrategies):
    """Stop strategies gated on the LLM's turn-completion verdict.

    The LLM is asked to begin every response with one of three markers:
    ✓ (complete), ○ (incomplete short), or ◐ (incomplete long). Only ✓
    finalizes the user turn; ○ / ◐ keep the turn open so the user can
    continue speaking and the LLM can re-evaluate later.

    Configuring strategies this way preserves the existing detector
    chain (defaults or user-supplied) for inference triggering and
    appends :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`
    as the finalizer. The detector strategies are wrapped with
    :func:`~pipecat.turns.user_stop.deferred` automatically so they fire
    only ``on_user_turn_inference_triggered`` and leave finalization to
    the LLM gate.

    Parameters:
        config: Optional configuration applied to the LLM via the
            ``filter_incomplete_user_turns`` setting. Customizes the
            turn-completion instructions, incomplete-turn timeouts, and
            re-prompts. If None, defaults from
            :class:`~pipecat.turns.user_turn_completion_mixin.UserTurnCompletionConfig`
            are used.

    Example::

        user_turn_strategies=FilterIncompleteUserTurnStrategies()

        # Custom detector chain:
        user_turn_strategies=FilterIncompleteUserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(...)],
        )

        # Custom completion config:
        user_turn_strategies=FilterIncompleteUserTurnStrategies(
            config=UserTurnCompletionConfig(
                incomplete_short_timeout=5.0,
                incomplete_long_timeout=10.0,
            ),
        )
    """

    config: UserTurnCompletionConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        # Defer the detector chain so it only fires inference-triggered,
        # then append the LLM gate as the sole finalizer.
        gated: list[BaseUserTurnStopStrategy] = [deferred(s) for s in self.stop or []]
        gated.append(LLMTurnCompletionUserTurnStopStrategy(config=self.config))
        self.stop = gated
