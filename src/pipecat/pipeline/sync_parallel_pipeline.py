#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from itertools import chain
from typing import List

from loguru import logger

from pipecat.frames.frames import ControlFrame, EndFrame, Frame, SystemFrame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.watchdog_queue import WatchdogQueue


@dataclass
class SyncFrame(ControlFrame):
    """This frame is used to know when the internal pipelines have finished."""

    pass


class SyncParallelPipelineSource(FrameProcessor):
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


class SyncParallelPipelineSink(FrameProcessor):
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


class SyncParallelPipeline(BasePipeline):
    def __init__(self, *args):
        super().__init__()

        if len(args) == 0:
            raise Exception(f"SyncParallelPipeline needs at least one argument")

        self._args = args
        self._sinks = []
        self._sources = []
        self._pipelines = []

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
                raise TypeError(f"SyncParallelPipeline argument {processors} is not a list")

            # We add a source at the beginning of the pipeline and a sink at the end.
            up_queue = asyncio.Queue()
            down_queue = asyncio.Queue()
            source = SyncParallelPipelineSource(up_queue)
            sink = SyncParallelPipelineSink(down_queue)

            # Create pipeline
            pipeline = Pipeline(processors)
            source.link(pipeline)
            pipeline.link(sink)
            self._pipelines.append(pipeline)

            # Keep track of sources and sinks. We also keep the output queue of
            # the source and the sinks so we can use it later.
            self._sources.append({"processor": source, "queue": down_queue})
            self._sinks.append({"processor": sink, "queue": up_queue})

        logger.debug(f"Finished creating {self} pipelines")

        await asyncio.gather(*[s["processor"].setup(setup) for s in self._sources])
        await asyncio.gather(*[p.setup(setup) for p in self._pipelines])
        await asyncio.gather(*[s["processor"].setup(setup) for s in self._sinks])

    async def cleanup(self):
        await super().cleanup()
        await asyncio.gather(*[s["processor"].cleanup() for s in self._sources])
        await asyncio.gather(*[p.cleanup() for p in self._pipelines])
        await asyncio.gather(*[s["processor"].cleanup() for s in self._sinks])

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # The last processor of each pipeline needs to be synchronous otherwise
        # this element won't work. Since, we know it should be synchronous we
        # push a SyncFrame. Since frames are ordered we know this frame will be
        # pushed after the synchronous processor has pushed its data allowing us
        # to synchrnonize all the internal pipelines by waiting for the
        # SyncFrame in all of them.
        async def wait_for_sync(
            obj, main_queue: asyncio.Queue, frame: Frame, direction: FrameDirection
        ):
            processor = obj["processor"]
            queue = obj["queue"]

            await processor.process_frame(frame, direction)

            if isinstance(frame, (SystemFrame, EndFrame)):
                new_frame = await queue.get()
                if isinstance(new_frame, (SystemFrame, EndFrame)):
                    await main_queue.put(new_frame)
                else:
                    while not isinstance(new_frame, (SystemFrame, EndFrame)):
                        await main_queue.put(new_frame)
                        queue.task_done()
                        new_frame = await queue.get()
            else:
                await processor.process_frame(SyncFrame(), direction)
                new_frame = await queue.get()
                while not isinstance(new_frame, SyncFrame):
                    await main_queue.put(new_frame)
                    queue.task_done()
                    new_frame = await queue.get()

        if direction == FrameDirection.UPSTREAM:
            # If we get an upstream frame we process it in each sink.
            await asyncio.gather(
                *[wait_for_sync(s, self._up_queue, frame, direction) for s in self._sinks]
            )
        elif direction == FrameDirection.DOWNSTREAM:
            # If we get a downstream frame we process it in each source.
            await asyncio.gather(
                *[wait_for_sync(s, self._down_queue, frame, direction) for s in self._sources]
            )

        seen_ids = set()
        while not self._up_queue.empty():
            frame = await self._up_queue.get()
            if frame.id not in seen_ids:
                await self.push_frame(frame, FrameDirection.UPSTREAM)
                seen_ids.add(frame.id)
            self._up_queue.task_done()

        seen_ids = set()
        while not self._down_queue.empty():
            frame = await self._down_queue.get()
            if frame.id not in seen_ids:
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)
                seen_ids.add(frame.id)
            self._down_queue.task_done()
