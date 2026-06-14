#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that finalizes on ``UserTurnInferenceCompletedFrame``."""

from pipecat.frames.frames import (
    Frame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy


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

    For LLM-completion-marker gating specifically, use the subclass
    :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`
    instead, which additionally pushes the ``LLMUpdateSettingsFrame``
    that enables the marker protocol on the LLM.

    If the producer never emits ``UserTurnInferenceCompletedFrame``, the
    controller's ``user_turn_stop_timeout`` watchdog finalizes the
    turn after no activity. Tune that timeout if your producer can
    take longer than the default to respond.

    Finalization is re-validated against live VAD: a completion frame
    is ignored while the user is acoustically speaking. The external
    verdict can arrive with latency (e.g. an LLM ``✓``), so the user
    may have resumed speaking by the time it lands; finalizing then
    would let the bot talk over them. When this happens the turn stays
    open and a later inference re-evaluates once the user is silent
    again. This mirrors the ``not self._user_speaking`` guard the
    controller already applies on its watchdog path.
    """

    def __init__(self, **kwargs):
        """Initialize the external user turn completion stop strategy.

        Args:
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self._user_speaking = False

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._user_speaking = False

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Fire ``on_user_turn_stopped`` on completion, gated on live VAD state."""
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
        elif isinstance(frame, UserTurnInferenceCompletedFrame):
            # Re-validate the verdict against live VAD. A completion that
            # resolves after the user resumed speaking is stale — keep the
            # turn open and let the next inference re-evaluate.
            if not self._user_speaking:
                await self.trigger_user_turn_finalized()
        return ProcessFrameResult.CONTINUE
