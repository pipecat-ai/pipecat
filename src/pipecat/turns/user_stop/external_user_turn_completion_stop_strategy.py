#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that finalizes on ``UserTurnInferenceCompletedFrame``."""

from pipecat.frames.frames import Frame, UserTurnInferenceCompletedFrame
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.deprecation import deprecated


@deprecated(
    "`ExternalUserTurnCompletionStopStrategy` is deprecated since 1.15.0 and will be removed "
    "in 2.0.0. Use `ExternalUserTurnStopStrategy` instead."
)
class ExternalUserTurnCompletionStopStrategy(BaseUserTurnStopStrategy):
    """Finalize the user turn whenever a ``UserTurnInferenceCompletedFrame`` arrives.

    Generic stop strategy for pipelines where some external component
    (LLM with completion markers, STT with built-in turn detection, a
    dedicated end-of-turn classifier, custom user code, etc.) judges
    when a turn is semantically complete and emits
    :class:`~pipecat.frames.frames.UserTurnInferenceCompletedFrame`.

    Pair this with one or more ``deferred(...)``-wrapped detector
    strategies that drive ``on_user_turn_inference_triggered`` but
    leave finalization to this strategy::

        stop=[
            deferred(TurnAnalyzerUserTurnStopStrategy(turn_analyzer=...)),
            ExternalUserTurnCompletionStopStrategy(),
        ]

    For LLM-completion-marker gating specifically, use
    :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`
    instead, which also turns the markers on in the LLM and re-prompts a user
    who goes quiet after an incomplete turn.

    A completion resolves with some latency (e.g. the LLM ``●`` arrives
    after the inference finishes), so the user may have resumed speaking
    in the meantime. The controller drops a finalization that arrives
    while the user is speaking, so a stale completion does not end the
    turn (and talk over the user); the turn stays open for the next
    inference to re-evaluate. That check lives in the controller, which
    holds the authoritative user-speaking state.

    If the producer never emits ``UserTurnInferenceCompletedFrame``, the
    controller's ``user_turn_stop_timeout`` watchdog finalizes the
    turn after no activity. Tune that timeout if your producer can
    take longer than the default to respond.

    .. deprecated:: 1.15.0
        Use :class:`~pipecat.turns.user_stop.ExternalUserTurnStopStrategy`
        instead, and push a
        :class:`~pipecat.frames.frames.ProposedUserStoppedSpeakingFrame` once
        the user is done. For the LLM's turn-completion markers,
        use :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`.
        Will be removed in 2.0.0.
    """

    @property
    def waits_for_inference(self) -> bool:
        """Decides on ``UserTurnInferenceCompletedFrame``, which the turn's inference produces."""
        return True

    async def process_frame(self, frame: Frame):
        """Fire ``on_user_turn_stopped`` whenever ``UserTurnInferenceCompletedFrame`` is seen."""
        if isinstance(frame, UserTurnInferenceCompletedFrame):
            await self.report_user_turn_stopped()
