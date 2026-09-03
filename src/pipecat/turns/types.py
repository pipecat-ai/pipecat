#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared types for user turn strategy frame processing."""

from dataclasses import dataclass
from enum import Enum


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
class Speculation:
    """A speculative inference a stop strategy has in flight.

    Produced from an eager end of turn: the turn is not over yet, so the
    inference runs against a provisional context and its response is held back
    until the turn is confirmed.

    Parameters:
        id: Identifies this speculation across the pipeline. The LLM service
            stamps it onto the response frames so the gate holding the response
            can tell which frames it owns.
        text: The user turn text the inference was run against.
    """

    id: str
    text: str
