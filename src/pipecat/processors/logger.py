#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger
from typing import Optional
logger = logger.opt(ansi=True)


class FrameLogger(FrameProcessor):
    def __init__(self, prefix="Frame", color: Optional[str] = None):
        super().__init__()
        self._prefix = prefix
        self._color = color

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        dir = "<" if direction is FrameDirection.UPSTREAM else ">"
        msg = f"{dir} {self._prefix}: {frame}"
        if self._color:
            msg = f"<{self._color}>{msg}</>"
        logger.debug(msg)

        await self.push_frame(frame, direction)
