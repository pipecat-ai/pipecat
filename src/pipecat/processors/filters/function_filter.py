#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Awaitable, Callable

from pipecat.frames.frames import Frame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FunctionFilter(FrameProcessor):
    def __init__(
        self,
        filter: Callable[[Frame], Awaitable[bool]],
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        super().__init__()
        self._filter = filter
        self._direction = direction

    #
    # Frame processor
    #

    # Ignore system frames and frames that are not following the direction of this gate
    def _should_passthrough_frame(self, frame, direction):
        return isinstance(frame, SystemFrame) or direction != self._direction

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        passthrough = self._should_passthrough_frame(frame, direction)
        allowed = await self._filter(frame)
        if passthrough or allowed:
            await self.push_frame(frame, direction)
