#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Null filter processor for blocking frame transmission.

This module provides a frame processor that blocks all frames except
system and end frames, useful for testing or temporarily stopping
frame flow in a pipeline.
"""

from pipecat.frames.frames import EndFrame, Frame, StopFrame, SystemFrame, WorkerFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class NullFilter(FrameProcessor):
    """A filter that blocks all frames except system and end frames.

    This processor acts as a null filter, preventing frames from passing
    through the pipeline while still allowing system frames and the frames
    that end or stop the pipeline (``EndFrame``, ``StopFrame`` and worker
    frames such as ``EndWorkerFrame``) to maintain proper pipeline operation.
    """

    def __init__(self, **kwargs):
        """Initialize the null filter.

        Args:
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, only allowing system and end frames through.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (SystemFrame, EndFrame, StopFrame, WorkerFrame)):
            await self.push_frame(frame, direction)
