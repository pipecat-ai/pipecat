#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FrameLogger(FrameProcessor):
    def __init__(self, prefix="Frame"):
        super().__init__()
        self._prefix = prefix

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        match direction:
            case FrameDirection.UPSTREAM:
                print(f"< {self._prefix}: {frame}")
            case FrameDirection.DOWNSTREAM:
                print(f"> {self._prefix}: {frame}")
        await self.push_frame(frame, direction)
