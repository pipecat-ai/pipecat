#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn start strategy configuration."""

from dataclasses import dataclass

from pipecat.classifiers.base_classifier import BaseClassifier
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    ClassifierUserTurnCompletionStopStrategy,
    EagerMatchPolicy,
    EagerUserTurnStopStrategy,
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


@dataclass
class ClassifierUserTurnStrategies(UserTurnStrategies):
    """Stop strategies gated on a classifier's turn-completion verdict.

    Keeps the detector chain (defaults or user-supplied) and wraps each
    detector in a
    :class:`~pipecat.turns.user_stop.ClassifierUserTurnCompletionStopStrategy`,
    so a detector firing asks the classifier whether the turn is complete
    instead of ending it. The LLM runs only when a turn is complete.

    Parameters:
        classifier: What decides whether a turn is complete. Required.
        context: The conversation, so the classifier sees the assistant's
            last message along with the user's turn.
        short_timeout: Seconds to keep a turn open after a ``short`` verdict.
        long_timeout: Seconds to keep a turn open after a ``long`` verdict.
        classification_timeout: Seconds to wait for the classifier before
            ending the turn without it.

    Example::

        user_turn_strategies=ClassifierUserTurnStrategies(
            classifier=JevClassifier(api_key=os.getenv("TYPESAFE_API_KEY")),
            context=context,
        )
    """

    classifier: BaseClassifier | None = None
    context: LLMContext | None = None
    short_timeout: float = 5.0
    long_timeout: float = 10.0
    classification_timeout: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.classifier is None:
            raise ValueError("ClassifierUserTurnStrategies needs a classifier")
        self.stop = [
            ClassifierUserTurnCompletionStopStrategy(
                s,
                classifier=self.classifier,
                context=self.context,
                short_timeout=self.short_timeout,
                long_timeout=self.long_timeout,
                classification_timeout=self.classification_timeout,
            )
            for s in self.stop or []
        ]


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
