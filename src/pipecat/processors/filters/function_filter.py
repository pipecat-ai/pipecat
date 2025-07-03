#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Function-based frame filtering for Pipecat pipelines.

This module provides a processor that filters frames based on a custom function,
allowing for flexible frame filtering logic in processing pipelines.
"""

from typing import Awaitable, Callable

from pipecat.frames.frames import EndFrame, Frame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FunctionFilter(FrameProcessor):
    """A frame processor that filters frames using a custom function.

    This processor allows frames to pass through based on the result of a
    user-provided filter function. System and end frames always pass through
    regardless of the filter result.
    """

    def __init__(
        self,
        filter: Callable[[Frame], Awaitable[bool]],
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        """Initialize the function filter.

        Args:
            filter: An async function that takes a Frame and returns True if the
                frame should pass through, False otherwise.
            direction: The direction to apply filtering. Only frames moving in
                this direction will be filtered. Defaults to DOWNSTREAM.
        """
        super().__init__()
        self._filter = filter
        self._direction = direction

    #
    # Frame processor
    #

    # Ignore system frames, end frames and frames that are not following the
    # direction of this gate
    def _should_passthrough_frame(self, frame, direction):
        """Check if a frame should pass through without filtering."""
        # Ignore system frames, end frames and frames that are not following the
        # direction of this gate
        return isinstance(frame, (SystemFrame, EndFrame)) or direction != self._direction

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame through the filter.

        Args:
            frame: The frame to process.
            direction: The direction the frame is moving in the pipeline.
        """
        await super().process_frame(frame, direction)

        passthrough = self._should_passthrough_frame(frame, direction)
        allowed = await self._filter(frame)
        if passthrough or allowed:
            await self.push_frame(frame, direction)
