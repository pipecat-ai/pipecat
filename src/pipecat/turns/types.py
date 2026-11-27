#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared types for user turn strategy frame processing."""

from dataclasses import dataclass
from enum import Enum

from pipecat.expressive.perception import PerceptionSignal


class ProcessFrameResult(Enum):
    """Result of processing a frame in a user turn strategy.

    Controls whether the strategy loop in the controller continues to the
    next strategy or stops early.

    Parameters:
        CONTINUE: Continue to the next strategy in the loop.
        STOP: Stop evaluating further strategies for this frame.

    .. deprecated:: 1.15.0
        Use :class:`~pipecat.turns.user_turn_strategies.StrategyMode` instead.
        Every strategy sees every frame, and a list whose strategies all have
        to agree, such as a wake phrase next to VAD, is ``StrategyMode.ALL``.
        Will be removed in 2.0.0.
    """

    CONTINUE = "continue"
    STOP = "stop"


@dataclass
class UserTurnSpeculation:
    """A speculative inference a stop strategy has in flight.

    Produced from an eager end of turn: the turn is not over yet, so the
    inference runs against a provisional context and its response is held back
    until the turn is confirmed.

    Parameters:
        text: The text the service predicted, covering the speech since it last
            committed a transcript.
        perception: How the user said it, as read at the eager end of turn,
            or None. Part of expressive conversation (alpha).
    """

    text: str
    perception: list[PerceptionSignal] | None = None


class InterruptionMode(Enum):
    """How a user turn start affects an in-progress bot response.

    Parameters:
        IMMEDIATE: Broadcast :class:`~pipecat.frames.frames.InterruptionFrame`
            as soon as the user turn starts.
        NONE: Do not interrupt the bot when the user turn starts.
        HOLD: Broadcast :class:`~pipecat.frames.frames.InterruptionHoldFrame`
            to pause output and buffer incoming bot frames until either a
            transcript arrives (committing an interruption) or the turn ends
            empty (resuming playback via
            :class:`~pipecat.frames.frames.InterruptionResumeFrame`).
    """

    IMMEDIATE = "immediate"
    NONE = "none"
    HOLD = "hold"


def resolve_interruption_mode(
    interruptions: InterruptionMode | None = None,
    enable_interruptions: bool | None = None,
) -> InterruptionMode:
    """Resolve an interruption mode from ``interruptions`` or ``enable_interruptions``.

    Args:
        interruptions: Explicit interruption mode, if provided.
        enable_interruptions: Legacy boolean flag mapped to
            ``InterruptionMode.IMMEDIATE`` / ``InterruptionMode.NONE``.

    Returns:
        The resolved :class:`InterruptionMode`, defaulting to
        ``InterruptionMode.IMMEDIATE`` when neither argument is specified.

    Raises:
        ValueError: If both ``interruptions`` and ``enable_interruptions`` are
            provided at the same time.
    """
    if interruptions is not None and enable_interruptions is not None:
        raise ValueError("Cannot specify both 'interruptions' and 'enable_interruptions'")
    if interruptions is not None:
        return interruptions
    if enable_interruptions is not None:
        return InterruptionMode.IMMEDIATE if enable_interruptions else InterruptionMode.NONE
    return InterruptionMode.IMMEDIATE


@dataclass
class UserTurnResumedParams:
    """Parameters emitted when the user speaks again in an open turn.

    Parameters:
        enable_interruptions: Whether the turn interrupts the bot, as decided
            when it started.
        inference_triggered: Whether the turn's inference was already
            triggered. The new speech voids the answer it started.
        interruption_mode: Interruption mode decided when the turn started.
            Synchronized with ``enable_interruptions`` when omitted.
    """

    enable_interruptions: bool
    inference_triggered: bool
    interruption_mode: InterruptionMode | None = None

    def __post_init__(self):
        """Keep ``enable_interruptions`` and ``interruption_mode`` consistent."""
        if self.interruption_mode is None:
            self.interruption_mode = (
                InterruptionMode.IMMEDIATE if self.enable_interruptions else InterruptionMode.NONE
            )
        else:
            self.enable_interruptions = self.interruption_mode != InterruptionMode.NONE
