#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gated frame aggregator for conditional frame accumulation.

This module provides a gated aggregator that accumulates frames based on
custom gate open/close functions, allowing for conditional frame buffering
and release in frame processing pipelines.
"""

from typing import List, Tuple

from loguru import logger

from pipecat.frames.frames import Frame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class GatedAggregator(FrameProcessor):
    """Accumulate frames, with custom functions to start and stop accumulation.

    Yields gate-opening frame before any accumulated frames, then ensuing frames
    until and not including the gate-closed frame. The aggregator maintains an
    internal gate state that controls whether frames are passed through immediately
    or accumulated for later release.
    """

    def __init__(
        self,
        gate_open_fn,
        gate_close_fn,
        start_open,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        """Initialize the gated aggregator.

        Args:
            gate_open_fn: Function that returns True when a frame should open the gate.
            gate_close_fn: Function that returns True when a frame should close the gate.
            start_open: Whether the gate should start in the open state.
            direction: The frame direction this aggregator operates on.
        """
        super().__init__()
        self._gate_open_fn = gate_open_fn
        self._gate_close_fn = gate_close_fn
        self._gate_open = start_open
        self._direction = direction
        self._accumulator: List[Tuple[Frame, FrameDirection]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames with gated accumulation logic.

        Args:
            frame: The frame to process.
            direction: The direction of the frame flow.
        """
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        # Ignore frames that are not following the direction of this gate.
        if direction != self._direction:
            await self.push_frame(frame, direction)
            return

        old_state = self._gate_open
        if self._gate_open:
            self._gate_open = not self._gate_close_fn(frame)
        else:
            self._gate_open = self._gate_open_fn(frame)

        if old_state != self._gate_open:
            state = "open" if self._gate_open else "closed"
            logger.debug(f"Gate is now {state} because of {frame}")

        if self._gate_open:
            await self.push_frame(frame, direction)
            for f, d in self._accumulator:
                await self.push_frame(f, d)
            self._accumulator = []
        else:
            self._accumulator.append((frame, direction))
