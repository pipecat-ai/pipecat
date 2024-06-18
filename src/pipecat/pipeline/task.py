#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from typing import AsyncIterable, Iterable

from pydantic import BaseModel

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, MetricsFrame, StartFrame, StopTaskFrame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.utils import obj_count, obj_id

from loguru import logger


class PipelineParams(BaseModel):
    allow_interruptions: bool = False
    enable_metrics: bool = False
    report_only_initial_ttfb: bool = False


class Source(FrameProcessor):

    def __init__(self, up_queue: asyncio.Queue):
        super().__init__()
        self._up_queue = up_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self._up_queue.put(frame)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)


class PipelineTask:

    def __init__(self, pipeline: BasePipeline, params: PipelineParams = PipelineParams()):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"

        self._pipeline = pipeline
        self._params = params
        self._finished = False

        self._down_queue = asyncio.Queue()
        self._up_queue = asyncio.Queue()

        self._source = Source(self._up_queue)
        self._source.link(pipeline)

    def has_finished(self):
        return self._finished

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
        await self._process_down_task
        await self._process_up_task

    async def run(self):
        self._process_up_task = asyncio.create_task(self._process_up_queue())
        self._process_down_task = asyncio.create_task(self._process_down_queue())
        await asyncio.gather(self._process_up_task, self._process_down_task)
        self._finished = True

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

    def _initial_metrics_frame(self) -> MetricsFrame:
        processors = self._pipeline.processors_with_metrics()
        ttfb = dict(zip([p.name for p in processors], [0] * len(processors)))
        return MetricsFrame(ttfb=ttfb)

    async def _process_down_queue(self):
        start_frame = StartFrame(
            allow_interruptions=self._params.allow_interruptions,
            enable_metrics=self._params.enable_metrics,
            report_only_initial_ttfb=self._params.report_only_initial_ttfb
        )
        await self._source.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        await self._source.process_frame(self._initial_metrics_frame(), FrameDirection.DOWNSTREAM)

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
        await self._process_up_task

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
