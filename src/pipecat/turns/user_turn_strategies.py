#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn start strategy configuration."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

from pydantic import BaseModel

from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    EagerMatchPolicy,
    EagerUserTurnStopStrategy,
    ExternalUserTurnStopStrategy,
    LLMTurnCompletionUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig

_S = TypeVar("_S")


class MergeMode(Enum):
    """How a bot's list of user turn strategies combines with a service's.

    Parameters:
        EXTEND: The strategies the bot gave are added to the service's list, or
            to the defaults when no service recommends any.
        REPLACE: The bot's list is used on its own.
    """

    EXTEND = "extend"
    REPLACE = "replace"


class StrategyMode(Enum):
    """How a list of user turn strategies reaches a decision.

    Parameters:
        ALL: Every strategy has to agree, and each one sees every frame.
        FIRST: The first strategy to decide wins.
    """

    ALL = "all"
    FIRST = "first"


class UserTurnStrategiesParams(BaseModel):
    """How the user turn strategies combine with a service's and decide.

    A service with its own turn detection, such as an STT service, can
    recommend user turn strategies. Only the bot sets these parameters: a
    service's recommendation contributes its lists.

    Parameters:
        start_merge_mode: How the bot's start strategies combine with the ones a
            service recommends. Defaults to ``MergeMode.REPLACE``.
        stop_merge_mode: How the bot's stop strategies combine with the ones a
            service recommends. Defaults to ``MergeMode.REPLACE``.
        start_mode: Whether the turn starts on the first start strategy that
            decides the user started, or once every start strategy has.
            Defaults to ``StrategyMode.FIRST``, so speech or a transcript
            starts a turn on its own. In a ``StrategyMode.ALL`` list every
            strategy is a condition the turn needs, so a fallback such as
            ``TranscriptionUserTurnStartStrategy``, which starts a turn when
            VAD misses speech, belongs in a ``StrategyMode.FIRST`` list.
        stop_mode: Whether the turn ends once every stop strategy has decided the
            user is done, or on the first one that does. A stop strategy that
            waits for the turn's inference, such as
            ``LLMTurnCompletionUserTurnStopStrategy``, decides on it in either
            mode. The user speaking again clears the decisions made so far.
            Defaults to ``StrategyMode.ALL``.
    """

    start_merge_mode: MergeMode = MergeMode.REPLACE
    stop_merge_mode: MergeMode = MergeMode.REPLACE
    start_mode: StrategyMode = StrategyMode.FIRST
    stop_mode: StrategyMode = StrategyMode.ALL

    @property
    def replaces_service_strategies(self) -> bool:
        """Whether the bot's strategies replace both of a service's lists."""
        return (
            self.start_merge_mode == MergeMode.REPLACE and self.stop_merge_mode == MergeMode.REPLACE
        )


@dataclass
class UserTurnStrategies:
    """Container for user turn start and stop strategies.

    If no strategies are specified, the following defaults are used:

        start: [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]
         stop: [TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)]

    A bot's strategies replace the ones a service recommends, unless
    :class:`UserTurnStrategiesParams` extends a list.

    Parameters:
        start: A list of user turn start strategies used to detect when
            the user starts speaking.
        stop: A list of user turn stop strategies used to decide when
            the user stops speaking.

    """

    start: list[BaseUserTurnStartStrategy] | None = None
    stop: list[BaseUserTurnStopStrategy] | None = None
    # What extending a service's lists adds: the strategies given, without the
    # defaults filled in for a list left out. None means the whole list.
    _start_extension: list[BaseUserTurnStartStrategy] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _stop_extension: list[BaseUserTurnStopStrategy] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        self._start_extension = list(self.start or [])
        self._stop_extension = list(self.stop or [])
        if not self.start:
            self.start = default_user_turn_start_strategies()
        if not self.stop:
            self.stop = default_user_turn_stop_strategies()

    def merge_with(
        self, recommended: "UserTurnStrategies | None", params: UserTurnStrategiesParams
    ) -> "UserTurnStrategies":
        """Combine these strategies with the ones a service recommends.

        Args:
            recommended: The strategies a service recommends, or None when no
                service recommends any, in which case the defaults stand in.
            params: How each list combines with the service's.

        Returns:
            The strategies to run.
        """
        return UserTurnStrategies(
            start=self._merge_list(
                self.start or [],
                self._start_extension,
                params.start_merge_mode,
                recommended.start if recommended else None,
                default_user_turn_start_strategies,
            ),
            stop=self._merge_list(
                self.stop or [],
                self._stop_extension,
                params.stop_merge_mode,
                recommended.stop if recommended else None,
                default_user_turn_stop_strategies,
            ),
        )

    @staticmethod
    def _merge_list(
        own: list[_S],
        extension: list[_S] | None,
        mode: MergeMode,
        recommended: list[_S] | None,
        defaults: Callable[[], list[_S]],
    ) -> list[_S]:
        """Merge one list with the service's, the defaults standing in for none."""
        if mode == MergeMode.REPLACE:
            return own
        added = own if extension is None else extension
        return (recommended if recommended is not None else defaults()) + added


@dataclass
class ExternalUserTurnStrategies(UserTurnStrategies):
    """Container for turn strategies driven by another component in the pipeline.

    Preconfigures :class:`UserTurnStrategies` with
    :class:`~pipecat.turns.user_start.ExternalUserTurnStartStrategy` and
    :class:`~pipecat.turns.user_stop.ExternalUserTurnStopStrategy`, so a service
    with its own turn detection — or a shared
    :class:`~pipecat.turns.user_turn_processor.UserTurnProcessor` — controls when
    user turns start and stop.

    What the aggregator emits depends on which signal drives the turn.
    ``ProposedUserStarted/StoppedSpeakingFrame`` leaves the decision here, so the
    aggregator pushes the turn frames and broadcasts interruptions.
    ``UserStarted/StoppedSpeakingFrame`` means the emitter already announced the
    turn, so the aggregator emits nothing and the parameter below doesn't apply.

    Parameters:
        enable_interruptions: Whether to broadcast an interruption when a
            proposal starts a turn. Services route their ``should_interrupt``
            setting here.

    """

    enable_interruptions: bool = True

    def __post_init__(self):
        self.start = [ExternalUserTurnStartStrategy(enable_interruptions=self.enable_interruptions)]
        self.stop = [ExternalUserTurnStopStrategy()]


@dataclass
class FilterIncompleteUserTurnStrategies(UserTurnStrategies):
    """Stop strategies gated on the LLM's turn-completion verdict.

    The LLM is asked to begin every response with one of three markers:
    ● (complete), ◐ (incomplete short), or ○ (incomplete long). Only ●
    finalizes the user turn; ◐ / ○ keep the turn open so the user can
    continue speaking and the LLM can re-evaluate later.

    Configuring strategies this way keeps the stop strategies (defaults or
    user-supplied) and adds
    :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`,
    which waits for the turn's inference. Once the other stop strategies
    decide the user stopped, the inference runs, and the turn ends once the
    LLM marks it complete. This works with either ``stop_mode``.

    With ``UserTurnStrategiesParams(stop_merge_mode=MergeMode.EXTEND)``, the
    LLM gate and any stop strategies given join the ones a service
    recommends instead, such as an STT service's own turn detection.

    Parameters:
        config: Optional configuration applied to the LLM via the
            ``filter_incomplete_user_turns`` setting. Customizes the
            turn-completion instructions, incomplete-turn timeouts, and
            re-prompts. If None, defaults from
            :class:`~pipecat.turns.user_turn_completion_mixin.UserTurnCompletionConfig`
            are used.

    Example::

        user_turn_strategies=FilterIncompleteUserTurnStrategies()

        # Custom stop strategies:
        user_turn_strategies=FilterIncompleteUserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(...)],
        )

        # On top of the turn detection a service recommends:
        user_turn_strategies=FilterIncompleteUserTurnStrategies(),
        user_turn_strategies_params=UserTurnStrategiesParams(
            stop_merge_mode=MergeMode.EXTEND,
        ),

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
        gate = LLMTurnCompletionUserTurnStopStrategy(config=self.config)
        self.stop = [*(self.stop or []), gate]
        # Extending a service's stop strategies adds the LLM gate and the stop
        # strategies given, never the default ones.
        self._stop_extension = [*(self._stop_extension or []), gate]


@dataclass
class EagerUserTurnStrategies(ExternalUserTurnStrategies):
    """Strategies for a service that predicts the end of a turn before committing.

    Answers the prediction while the turn is still open, so the gap before the
    committed end of turn is spent generating a response instead of waiting for
    one. The response is discarded if the user resumes speaking or the committed
    transcript differs from the eager one. See
    :class:`~pipecat.turns.user_stop.EagerUserTurnStopStrategy`.

    The response is held by the LLM service until the turn is confirmed, so no
    extra processor is needed in the pipeline.

    The service owns turn detection here, so this replaces the detector chain
    rather than extending it: a local detector running alongside would trigger a
    second inference for the same turn.

    Parameters:
        match_policy: Decides whether the committed transcript is close enough to
            the eager one to keep the speculative response. Defaults to
            :class:`~pipecat.turns.user_stop.NormalizedMatch`, since services
            commonly format the transcript they commit (capitalization,
            punctuation) while leaving the eager one raw. Pass
            :class:`~pipecat.turns.user_stop.ExactMatch` to require the two to
            be identical.
        speculation_timeout: Seconds a prediction may go unresolved before it
            is withdrawn and the turn falls back to answering the committed
            transcript. Guards against a service that stops sending turn signals
            mid-speculation, which would otherwise leave the response held and
            the bot silent.
        enable_interruptions: Whether to broadcast an interruption when a
            proposal starts a turn. Services route their ``should_interrupt``
            setting here.

    Example::

        user_turn_strategies=EagerUserTurnStrategies()

        # Require the committed transcript to match the eager one exactly:
        user_turn_strategies=EagerUserTurnStrategies(match_policy=ExactMatch())
    """

    match_policy: EagerMatchPolicy | None = None
    speculation_timeout: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        self.stop = [
            EagerUserTurnStopStrategy(
                match_policy=self.match_policy,
                speculation_timeout=self.speculation_timeout,
            )
        ]


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
