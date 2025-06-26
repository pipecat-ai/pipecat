#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from itertools import chain
from typing import Awaitable, Callable, Dict, List

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    SystemFrame,
)
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.watchdog_queue import WatchdogQueue


class ParallelPipelineSource(FrameProcessor):
    def __init__(
        self,
        upstream_queue: asyncio.Queue,
        push_frame_func: Callable[[Frame, FrameDirection], Awaitable[None]],
    ):
        super().__init__()
        self._up_queue = upstream_queue
        self._push_frame_func = push_frame_func

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                if isinstance(frame, SystemFrame):
                    await self._push_frame_func(frame, direction)
                else:
                    await self._up_queue.put(frame)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)


class ParallelPipelineSink(FrameProcessor):
    def __init__(
        self,
        downstream_queue: asyncio.Queue,
        push_frame_func: Callable[[Frame, FrameDirection], Awaitable[None]],
    ):
        super().__init__()
        self._down_queue = downstream_queue
        self._push_frame_func = push_frame_func

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self.push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                if isinstance(frame, SystemFrame):
                    await self._push_frame_func(frame, direction)
                else:
                    await self._down_queue.put(frame)


class ParallelPipeline(BasePipeline):
    def __init__(self, *args):
        super().__init__()

        if len(args) == 0:
            raise Exception(f"ParallelPipeline needs at least one argument")

        self._args = args
        self._sources = []
        self._sinks = []
        self._pipelines = []

        self._seen_ids = set()
        self._endframe_counter: Dict[int, int] = {}

        self._up_task = None
        self._down_task = None

    #
    # BasePipeline
    #

    def processors_with_metrics(self) -> List[FrameProcessor]:
        return list(chain.from_iterable(p.processors_with_metrics() for p in self._pipelines))

    #
    # Frame processor
    #

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)

        self._up_queue = WatchdogQueue(setup.task_manager)
        self._down_queue = WatchdogQueue(setup.task_manager)

        logger.debug(f"Creating {self} pipelines")
        for processors in self._args:
            if not isinstance(processors, list):
                raise TypeError(f"ParallelPipeline argument {processors} is not a list")

            # We will add a source before the pipeline and a sink after.
            source = ParallelPipelineSource(self._up_queue, self._parallel_push_frame)
            sink = ParallelPipelineSink(self._down_queue, self._parallel_push_frame)
            self._sources.append(source)
            self._sinks.append(sink)

            # Create pipeline
            pipeline = Pipeline(processors)
            source.link(pipeline)
            pipeline.link(sink)
            self._pipelines.append(pipeline)

        logger.debug(f"Finished creating {self} pipelines")

        await asyncio.gather(*[s.setup(setup) for s in self._sources])
        await asyncio.gather(*[p.setup(setup) for p in self._pipelines])
        await asyncio.gather(*[s.setup(setup) for s in self._sinks])

    async def cleanup(self):
        await super().cleanup()
        await asyncio.gather(*[s.cleanup() for s in self._sources])
        await asyncio.gather(*[p.cleanup() for p in self._pipelines])
        await asyncio.gather(*[s.cleanup() for s in self._sinks])

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            self._endframe_counter[frame.id] = len(self._pipelines)
        elif isinstance(frame, CancelFrame):
            await self._cancel()

        if direction == FrameDirection.UPSTREAM:
            # If we get an upstream frame we process it in each sink.
            await asyncio.gather(*[s.queue_frame(frame, direction) for s in self._sinks])
        elif direction == FrameDirection.DOWNSTREAM:
            # If we get a downstream frame we process it in each source.
            await asyncio.gather(*[s.queue_frame(frame, direction) for s in self._sources])

        # Handle interruptions after everything has been cancelled.
        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption()
        # Wait for tasks to finish.
        elif isinstance(frame, EndFrame):
            await self._stop()

    async def _start(self, frame: StartFrame):
        await self._create_tasks()

    async def _stop(self):
        if self._up_task:
            # The up task doesn't receive an EndFrame, so we just cancel it.
            await self.cancel_task(self._up_task)
            self._up_task = None

        if self._down_task:
            # The down tasks waits for the last EndFrame sent by the internal
            # pipelines.
            await self._down_task
            self._down_task = None

    async def _cancel(self):
        if self._up_task:
            await self.cancel_task(self._up_task)
            self._up_task = None
        if self._down_task:
            await self.cancel_task(self._down_task)
            self._down_task = None

    async def _create_tasks(self):
        if not self._up_task:
            self._up_task = self.create_task(self._process_up_queue())
        if not self._down_task:
            self._down_task = self.create_task(self._process_down_queue())

    async def _drain_queues(self):
        while not self._up_queue.empty:
            await self._up_queue.get()
        while not self._down_queue.empty:
            await self._down_queue.get()

    async def _handle_interruption(self):
        await self._cancel()
        await self._drain_queues()
        await self._create_tasks()

    async def _parallel_push_frame(self, frame: Frame, direction: FrameDirection):
        if frame.id not in self._seen_ids:
            self._seen_ids.add(frame.id)
            await self.push_frame(frame, direction)

    async def _process_up_queue(self):
        while True:
            frame = await self._up_queue.get()
            await self._parallel_push_frame(frame, FrameDirection.UPSTREAM)
            self._up_queue.task_done()

    async def _process_down_queue(self):
        running = True
        while running:
            frame = await self._down_queue.get()

            endframe_counter = self._endframe_counter.get(frame.id, 0)

            # If we have a counter, decrement it.
            if endframe_counter > 0:
                self._endframe_counter[frame.id] -= 1
                endframe_counter = self._endframe_counter[frame.id]

            # If we don't have a counter or we reached 0, push the frame.
            if endframe_counter == 0:
                await self._parallel_push_frame(frame, FrameDirection.DOWNSTREAM)

            running = not (endframe_counter == 0 and isinstance(frame, EndFrame))

            self._down_queue.task_done()
