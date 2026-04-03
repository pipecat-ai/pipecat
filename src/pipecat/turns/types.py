#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared result type for user turn strategy frame processing."""

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
