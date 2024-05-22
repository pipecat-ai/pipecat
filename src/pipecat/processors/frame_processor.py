#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from asyncio import AbstractEventLoop
from enum import Enum

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.utils.utils import obj_count, obj_id

from loguru import logger


class FrameDirection(Enum):
    DOWNSTREAM = 1
    UPSTREAM = 2


class FrameProcessor:

    def __init__(self):
        self.id: int = obj_id()
        self.name = f"{self.__class__.__name__}#{obj_count(self)}"
        self._prev: "FrameProcessor" | None = None
        self._next: "FrameProcessor" | None = None
        self._loop: AbstractEventLoop = asyncio.get_running_loop()

    async def cleanup(self):
        pass

    def link(self, processor: 'FrameProcessor'):
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")

    def get_event_loop(self) -> AbstractEventLoop:
        return self._loop

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        pass

    async def push_error(self, error: ErrorFrame):
        await self.push_frame(error, FrameDirection.UPSTREAM)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM and self._next:
            logger.trace(f"Pushing {frame} from {self} to {self._next}")
            await self._next.process_frame(frame, direction)
        elif direction == FrameDirection.UPSTREAM and self._prev:
            logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
            await self._prev.process_frame(frame, direction)

    def __str__(self):
        return self.name
