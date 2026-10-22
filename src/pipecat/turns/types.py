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
