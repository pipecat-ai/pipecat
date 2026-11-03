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
        text: The user turn text the inference was run against.
        perception: How the user said it, as read at the eager end of turn,
            or None. Part of expressive conversation (alpha).
    """

    text: str
    perception: list[PerceptionSignal] | None = None


@dataclass
class UserTurnResumedParams:
    """Parameters emitted when the user speaks again in an open turn.

    Parameters:
        enable_interruptions: Whether the turn interrupts the bot, as decided
            when it started.
        inference_triggered: Whether the turn's inference was already
            triggered. The new speech voids the answer it started.
    """

    enable_interruptions: bool
    inference_triggered: bool
