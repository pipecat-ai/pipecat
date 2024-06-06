#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from enum import Enum

from pipecat.frames.frames import ErrorFrame, Frame, StartFrame
from pipecat.utils.utils import obj_count, obj_id

from loguru import logger


class FrameDirection(Enum):
    DOWNSTREAM = 1
    UPSTREAM = 2


class FrameProcessor:

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self.id: int = obj_id()
        self.name = f"{self.__class__.__name__}#{obj_count(self)}"
        self._prev: "FrameProcessor" | None = None
        self._next: "FrameProcessor" | None = None
        self._loop: asyncio.AbstractEventLoop = loop or asyncio.get_running_loop()

        # Properties
        self._allow_interruptions = False
        self._enable_metrics = False

    @property
    def allow_interruptions(self):
        return self._allow_interruptions

    @property
    def enable_metrics(self):
        return self._enable_metrics

    async def cleanup(self):
        pass

    def link(self, processor: 'FrameProcessor'):
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            self._allow_interruptions = frame.allow_interruptions
            self._enable_metrics = frame.enable_metrics

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
