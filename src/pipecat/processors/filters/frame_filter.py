#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame filtering processor for the Pipecat framework."""

from pipecat.frames.frames import EndFrame, Frame, StopFrame, SystemFrame, WorkerFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FrameFilter(FrameProcessor):
    """A frame processor that filters frames based on their types.

    This processor acts as a selective gate in the pipeline, allowing only
    frames of specified types to pass through. System frames, and the frames
    that end or stop the pipeline (``EndFrame``, ``StopFrame`` and worker
    frames such as ``EndWorkerFrame``), always pass through so the pipeline can
    still finish.
    """

    def __init__(self, types: tuple[type[Frame], ...]):
        """Initialize the frame filter.

        Args:
            types: Tuple of frame types that should be allowed to pass through
                   the filter. All other frame types (except system frames
                   and the frames that end or stop the pipeline) will be blocked.
        """
        super().__init__()
        self._types = types

    #
    # Frame processor
    #

    def _should_passthrough_frame(self, frame):
        """Determine if a frame should pass through the filter."""
        if isinstance(frame, self._types):
            return True

        return isinstance(frame, (EndFrame, StopFrame, WorkerFrame, SystemFrame))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame and conditionally pass it through.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if self._should_passthrough_frame(frame):
            await self.push_frame(frame, direction)
