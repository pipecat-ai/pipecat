#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Awaitable, Callable

from pipecat.frames.frames import Frame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FunctionFilter(FrameProcessor):

    def __init__(self, filter: Callable[[Frame], Awaitable[bool]]):
        super().__init__()
        self._filter = filter

    #
    # Frame processor
    #

    def _should_passthrough_frame(self, frame):
        return isinstance(frame, SystemFrame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        passthrough = self._should_passthrough_frame(frame)
        allowed = await self._filter(frame)
        if passthrough or allowed:
            await self.push_frame(frame, direction)
