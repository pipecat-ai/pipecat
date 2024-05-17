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

    def __init__(self, pipeline: FrameProcessor, allow_interruptions=False):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"

        self._pipeline = pipeline
        self._allow_interruptions = allow_interruptions

        self._down_queue = asyncio.Queue()
        self._up_queue = asyncio.Queue()

        self._source = Source(self._up_queue)
        self._source.link(pipeline)

    async def stop_when_done(self):
        logger.debug(f"Task {self} scheduled to stop when done")
        await self.queue_frame(EndFrame())

    async def cancel(self):
        logger.debug(f"Canceling pipeline task {self}")
        # Make sure everything is cleaned up downstream. This is sent
        # out-of-band from the main streaming task which is what we want since
        # we want to cancel right away.
        await self._source.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
        self._process_down_task.cancel()
        self._process_up_task.cancel()

    async def run(self):
        self._process_up_task = asyncio.create_task(self._process_up_queue())
        self._process_down_task = asyncio.create_task(self._process_down_queue())
        await asyncio.gather(self._process_up_task, self._process_down_task)

    async def queue_frame(self, frame: Frame):
        await self._down_queue.put(frame)

    async def queue_frames(self, frames: Iterable[Frame] | AsyncIterable[Frame]):
        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                await self.queue_frame(frame)
        elif isinstance(frames, Iterable):
            for frame in frames:
                await self.queue_frame(frame)
        else:
            raise Exception("Frames must be an iterable or async iterable")

    async def _process_down_queue(self):
        await self._source.process_frame(
            StartFrame(allow_interruptions=self._allow_interruptions), FrameDirection.DOWNSTREAM)
        running = True
        should_cleanup = True
        while running:
            try:
                frame = await self._down_queue.get()
                await self._source.process_frame(frame, FrameDirection.DOWNSTREAM)
                running = not (isinstance(frame, StopTaskFrame) or isinstance(frame, EndFrame))
                should_cleanup = not isinstance(frame, StopTaskFrame)
                self._down_queue.task_done()
            except asyncio.CancelledError:
                break
        # Cleanup only if we need to.
        if should_cleanup:
            await self._source.cleanup()
            await self._pipeline.cleanup()
        # We just enqueue None to terminate the task gracefully.
        self._process_up_task.cancel()

    async def _process_up_queue(self):
        while True:
            try:
                frame = await self._up_queue.get()
                if isinstance(frame, ErrorFrame):
                    logger.error(f"Error running app: {frame.error}")
                    await self.queue_frame(CancelFrame())
                self._up_queue.task_done()
            except asyncio.CancelledError:
                break

    def __str__(self):
        return self.name
