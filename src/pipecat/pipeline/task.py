#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from typing import AsyncIterable, Iterable

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, StartFrame, StopTaskFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.utils import obj_count, obj_id

from loguru import logger


class Source(FrameProcessor):

    def __init__(self, up_queue: asyncio.Queue):
        super().__init__()
        self._up_queue = up_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        match direction:
            case FrameDirection.UPSTREAM:
                await self._up_queue.put(frame)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)


class PipelineTask:

    def __init__(self, pipeline: FrameProcessor):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"

        self._pipeline = pipeline

        self._task_queue = asyncio.Queue()
        self._up_queue = asyncio.Queue()

        self._source = Source(self._up_queue)
        self._source.link(pipeline)

    async def stop_when_done(self):
        logger.debug(f"Task {self} scheduled to stop when done")
        await self.queue_frame(EndFrame())

    async def cancel(self):
        logger.debug(f"Canceling pipeline task {self}")
        await self.queue_frame(CancelFrame())

    async def run(self):
        await asyncio.gather(self._process_task_queue(), self._process_up_queue())

    async def queue_frame(self, frame: Frame):
        await self._task_queue.put(frame)

    async def queue_frames(self, frames: Iterable[Frame] | AsyncIterable[Frame]):
        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                await self.queue_frame(frame)
        elif isinstance(frames, Iterable):
            for frame in frames:
                await self.queue_frame(frame)
        else:
            raise Exception("Frames must be an iterable or async iterable")

    async def _process_task_queue(self):
        await self._source.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        running = True
        while running:
            frame = await self._task_queue.get()
            await self._source.process_frame(frame, FrameDirection.DOWNSTREAM)
            self._task_queue.task_done()
            running = not (isinstance(frame, StopTaskFrame) or
                           isinstance(frame, CancelFrame) or
                           isinstance(frame, EndFrame))
        # We just enqueue None to terminate the task.
        await self._up_queue.put(None)

    async def _process_up_queue(self):
        running = True
        while running:
            frame = await self._up_queue.get()
            if frame:
                if isinstance(frame, ErrorFrame):
                    logger.error(f"Error running app: {frame.error}")
                    await self.queue_frame(CancelFrame())
            self._up_queue.task_done()
            running = frame is not None

    def __str__(self):
        return self.name
