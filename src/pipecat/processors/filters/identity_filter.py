#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class IdentityFilter(FrameProcessor):
    """A pass-through filter that forwards all frames without modification.

    This filter acts as a transparent passthrough, allowing all frames to flow
    through unchanged. It can be useful when testing `ParallelPipeline` to
    create pipelines that pass through frames (no frames should be repeated).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame by passing it through unchanged."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
