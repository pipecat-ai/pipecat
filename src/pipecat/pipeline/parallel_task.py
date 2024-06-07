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
from pipecat.frames.frames import Frame

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


class ParallelTask(BasePipeline):
    def __init__(self, *args):
        super().__init__()

        if len(args) == 0:
            raise Exception(f"ParallelTask needs at least one argument")

        self._sinks = []
        self._pipelines = []

        self._up_queue = asyncio.Queue()
        self._down_queue = asyncio.Queue()

        logger.debug(f"Creating {self} pipelines")
        for processors in args:
            if not isinstance(processors, list):
                raise TypeError(f"ParallelTask argument {processors} is not a list")

            # We add a source at the beginning of the pipeline and a sink at the end.
            source = Source(self._up_queue)
            sink = Sink(self._down_queue)
            processors: List[FrameProcessor] = [source] + processors
            processors.append(sink)

            # Keep track of sinks. We access the source through the pipeline.
            self._sinks.append(sink)

            # Create pipeline
            pipeline = Pipeline(processors)
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            # If we get an upstream frame we process it in each sink.
            await asyncio.gather(*[s.process_frame(frame, direction) for s in self._sinks])
        elif direction == FrameDirection.DOWNSTREAM:
            # If we get a downstream frame we process it in each source (using the pipeline).
            await asyncio.gather(*[p.process_frame(frame, direction) for p in self._pipelines])

        seen_ids = set()
        while not self._up_queue.empty():
            frame = await self._up_queue.get()
            if frame and frame.id not in seen_ids:
                await self.push_frame(frame, FrameDirection.UPSTREAM)
                seen_ids.add(frame.id)
            self._up_queue.task_done()

        seen_ids = set()
        while not self._down_queue.empty():
            frame = await self._down_queue.get()
            if frame and frame.id not in seen_ids:
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)
                seen_ids.add(frame.id)
            self._down_queue.task_done()
