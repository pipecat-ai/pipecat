#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from typing import AsyncIterable, Iterable

from pydantic import BaseModel

from pipecat.clocks.base_clock import BaseClock
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    CancelTaskFrame,
    EndFrame,
    EndTaskFrame,
    ErrorFrame,
    Frame,
    MetricsFrame,
    StartFrame,
    StopTaskFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData, ProcessingMetricsData
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.utils import obj_count, obj_id

from loguru import logger


class PipelineParams(BaseModel):
    allow_interruptions: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    send_initial_empty_metrics: bool = True
    report_only_initial_ttfb: bool = False


class Source(FrameProcessor):
    def __init__(self, up_queue: asyncio.Queue):
        super().__init__()
        self._up_queue = up_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self._handle_upstream_frame(frame)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)

    async def _handle_upstream_frame(self, frame: Frame):
        if isinstance(frame, EndTaskFrame):
            # Tell the task we should end nicely.
            await self._up_queue.put(EndTaskFrame())
        elif isinstance(frame, CancelTaskFrame):
            # Tell the task we should end right away.
            await self._up_queue.put(CancelTaskFrame())
        elif isinstance(frame, ErrorFrame):
            logger.error(f"Error running app: {frame}")
            if frame.fatal:
                # Cancel all tasks downstream.
                await self.push_frame(CancelFrame())
                # Tell the task we should stop.
                await self._up_queue.put(StopTaskFrame())


class Sink(FrameProcessor):
    def __init__(self, down_queue: asyncio.Queue):
        super().__init__()
        self._down_queue = down_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We really just want to know when the EndFrame reached the sink.
        if isinstance(frame, EndFrame):
            await self._down_queue.put(frame)


class PipelineTask:
    def __init__(
        self,
        pipeline: BasePipeline,
        params: PipelineParams = PipelineParams(),
        clock: BaseClock = SystemClock(),
    ):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"

        self._pipeline = pipeline
        self._clock = clock
        self._params = params
        self._finished = False

        self._up_queue = asyncio.Queue()
        self._down_queue = asyncio.Queue()
        self._push_queue = asyncio.Queue()

        self._source = Source(self._up_queue)
        self._source.link(pipeline)

        self._sink = Sink(self._down_queue)
        pipeline.link(self._sink)

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
        await self._source.push_frame(CancelFrame())
        self._process_push_task.cancel()
        self._process_up_task.cancel()
        await self._process_push_task
        await self._process_up_task

    async def run(self):
        self._process_up_task = asyncio.create_task(self._process_up_queue())
        self._process_push_task = asyncio.create_task(self._process_push_queue())
        await asyncio.gather(self._process_up_task, self._process_push_task)
        self._finished = True

    async def queue_frame(self, frame: Frame):
        await self._push_queue.put(frame)

    async def queue_frames(self, frames: Iterable[Frame] | AsyncIterable[Frame]):
        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                await self.queue_frame(frame)
        elif isinstance(frames, Iterable):
            for frame in frames:
                await self.queue_frame(frame)

    def _initial_metrics_frame(self) -> MetricsFrame:
        processors = self._pipeline.processors_with_metrics()
        data = []
        for p in processors:
            data.append(TTFBMetricsData(processor=p.name, value=0.0))
            data.append(ProcessingMetricsData(processor=p.name, value=0.0))
        return MetricsFrame(data=data)

    async def _process_push_queue(self):
        self._clock.start()

        start_frame = StartFrame(
            allow_interruptions=self._params.allow_interruptions,
            enable_metrics=self._params.enable_metrics,
            enable_usage_metrics=self._params.enable_usage_metrics,
            report_only_initial_ttfb=self._params.report_only_initial_ttfb,
            clock=self._clock,
        )
        await self._source.process_frame(start_frame, FrameDirection.DOWNSTREAM)

        if self._params.enable_metrics and self._params.send_initial_empty_metrics:
            await self._source.process_frame(
                self._initial_metrics_frame(), FrameDirection.DOWNSTREAM
            )

        running = True
        should_cleanup = True
        while running:
            try:
                frame = await self._push_queue.get()
                await self._source.process_frame(frame, FrameDirection.DOWNSTREAM)
                if isinstance(frame, EndFrame):
                    await self._wait_for_endframe()
                running = not isinstance(frame, (StopTaskFrame, EndFrame))
                should_cleanup = not isinstance(frame, StopTaskFrame)
                self._push_queue.task_done()
            except asyncio.CancelledError:
                break
        # Cleanup only if we need to.
        if should_cleanup:
            await self._source.cleanup()
            await self._pipeline.cleanup()
        # We just enqueue None to terminate the task gracefully.
        self._process_up_task.cancel()
        await self._process_up_task

    async def _wait_for_endframe(self):
        # NOTE(aleix): the Sink element just pushes EndFrames to the down queue,
        # so just wait for it. In the future we might do something else here,
        # but for now this is fine.
        await self._down_queue.get()

    async def _process_up_queue(self):
        while True:
            try:
                frame = await self._up_queue.get()
                if isinstance(frame, EndTaskFrame):
                    await self.queue_frame(EndFrame())
                elif isinstance(frame, CancelTaskFrame):
                    await self.queue_frame(CancelFrame())
                elif isinstance(frame, StopTaskFrame):
                    await self.queue_frame(StopTaskFrame())
                self._up_queue.task_done()
            except asyncio.CancelledError:
                break

    def __str__(self):
        return self.name
