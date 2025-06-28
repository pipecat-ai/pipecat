#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Identity filter for transparent frame passthrough.

This module provides a simple passthrough filter that forwards all frames
without modification, useful for testing and pipeline composition.
"""

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class IdentityFilter(FrameProcessor):
    """A pass-through filter that forwards all frames without modification.

    This filter acts as a transparent passthrough, allowing all frames to flow
    through unchanged. It can be useful when testing `ParallelPipeline` to
    create pipelines that pass through frames (no frames should be repeated).
    """

    def __init__(self, **kwargs):
        """Initialize the identity filter.

        Args:
            **kwargs: Additional arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame by passing it through unchanged.

        Args:
            frame: The frame to process and forward.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
