#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from itertools import chain
from typing import List

from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame

from loguru import logger


class Source(FrameProcessor):
    def __init__(self, upstream_queue: asyncio.Queue):
        super().__init__()
        self._up_queue = upstream_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self._up_queue.put(frame)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    def __init__(self, downstream_queue: asyncio.Queue):
        super().__init__()
        self._down_queue = downstream_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self.push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                await self._down_queue.put(frame)


class ParallelPipeline(BasePipeline):
    def __init__(self, *args):
        super().__init__()

        if len(args) == 0:
            raise Exception(f"ParallelPipeline needs at least one argument")

        self._sources = []
        self._sinks = []

        self._up_queue = asyncio.Queue()
        self._down_queue = asyncio.Queue()
        self._up_task: asyncio.Task | None = None
        self._down_task: asyncio.Task | None = None

        self._pipelines = []

        logger.debug(f"Creating {self} pipelines")
        for processors in args:
            if not isinstance(processors, list):
                raise TypeError(f"ParallelPipeline argument {processors} is not a list")

            # We will add a source before the pipeline and a sink after.
            source = Source(self._up_queue)
            sink = Sink(self._down_queue)
            self._sources.append(source)
            self._sinks.append(sink)

            # Create pipeline
            pipeline = Pipeline(processors)
            source.link(pipeline)
            pipeline.link(sink)
            self._pipelines.append(pipeline)

        logger.debug(f"Finished creating {self} pipelines")

    #
    # BasePipeline
    #

    def processors_with_metrics(self) -> List[FrameProcessor]:
        return list(chain.from_iterable(p.processors_with_metrics() for p in self._pipelines))

    #
    # Frame processor
    #

    async def cleanup(self):
        await asyncio.gather(*[p.cleanup() for p in self._pipelines])

    async def _start_tasks(self):
        loop = self.get_event_loop()
        self._up_task = loop.create_task(self._process_up_queue())
        self._down_task = loop.create_task(self._process_down_queue())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start_tasks()

        if direction == FrameDirection.UPSTREAM:
            # If we get an upstream frame we process it in each sink.
            await asyncio.gather(*[s.process_frame(frame, direction) for s in self._sinks])
        elif direction == FrameDirection.DOWNSTREAM:
            # If we get a downstream frame we process it in each source.
            # TODO(aleix): We are creating task for each frame. For real-time
            # video/audio this might be too slow. We should use an already
            # created task instead.
            await asyncio.gather(*[s.process_frame(frame, direction) for s in self._sources])

        # If we get an EndFrame we stop our queue processing tasks and wait on
        # all the pipelines to finish.
        if isinstance(frame, (CancelFrame, EndFrame)):
            # Use None to indicate when queues should be done processing.
            await self._up_queue.put(None)
            await self._down_queue.put(None)
            if self._up_task:
                await self._up_task
            if self._down_task:
                await self._down_task

    async def _process_up_queue(self):
        running = True
        seen_ids = set()
        while running:
            frame = await self._up_queue.get()
            if frame and frame.id not in seen_ids:
                await self.push_frame(frame, FrameDirection.UPSTREAM)
                seen_ids.add(frame.id)
            running = frame is not None
            self._up_queue.task_done()

    async def _process_down_queue(self):
        running = True
        seen_ids = set()
        while running:
            frame = await self._down_queue.get()
            if frame and frame.id not in seen_ids:
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)
                seen_ids.add(frame.id)
            running = frame is not None
            self._down_queue.task_done()
