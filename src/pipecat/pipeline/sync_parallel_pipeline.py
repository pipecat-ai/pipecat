#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Synchronous parallel pipeline implementation for concurrent frame processing.

This module provides a pipeline that processes frames through multiple parallel
pipelines simultaneously, synchronizing their output to maintain frame ordering
and prevent duplicate processing.
"""

import asyncio
from dataclasses import dataclass
from itertools import chain
from typing import List

from loguru import logger

from pipecat.frames.frames import ControlFrame, EndFrame, Frame, SystemFrame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup


@dataclass
class SyncFrame(ControlFrame):
    """Control frame used to synchronize parallel pipeline processing.

    This frame is sent through parallel pipelines to determine when the
    internal pipelines have finished processing a batch of frames.
    """

    pass


class SyncParallelPipelineSource(FrameProcessor):
    """Source processor for synchronous parallel pipeline processing.

    Routes frames to parallel pipelines and collects upstream responses
    for synchronization purposes.
    """

    def __init__(self, upstream_queue: asyncio.Queue):
        """Initialize the sync parallel pipeline source.

        Args:
            upstream_queue: Queue for collecting upstream frames from the pipeline.
        """
        super().__init__(enable_direct_mode=True)
        self._up_queue = upstream_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and route them based on direction.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self._up_queue.put(frame)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)


class SyncParallelPipelineSink(FrameProcessor):
    """Sink processor for synchronous parallel pipeline processing.

    Collects downstream frames from parallel pipelines and routes
    upstream frames back through the pipeline.
    """

    def __init__(self, downstream_queue: asyncio.Queue):
        """Initialize the sync parallel pipeline sink.

        Args:
            downstream_queue: Queue for collecting downstream frames from the pipeline.
        """
        super().__init__(enable_direct_mode=True)
        self._down_queue = downstream_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and route them based on direction.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self.push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                await self._down_queue.put(frame)


class SyncParallelPipeline(BasePipeline):
    """Pipeline that processes frames through multiple parallel pipelines synchronously.

    Creates multiple parallel processing paths that all receive the same input frames
    and produces synchronized output. Each parallel path is a separate pipeline that
    processes frames independently, with synchronization points to ensure consistent
    ordering and prevent duplicate frame processing.

    The pipeline uses SyncFrame control frames to coordinate between parallel paths
    and ensure all paths have completed processing before moving to the next frame.
    """

    def __init__(self, *args):
        """Initialize the synchronous parallel pipeline.

        Args:
            *args: Variable number of processor lists, each representing a parallel pipeline path.
                   Each argument should be a list of FrameProcessor instances.

        Raises:
            Exception: If no arguments are provided.
            TypeError: If any argument is not a list of processors.
        """
        super().__init__()

        if len(args) == 0:
            raise Exception(f"SyncParallelPipeline needs at least one argument")

        self._sinks = []
        self._sources = []
        self._pipelines = []

        self._up_queue = asyncio.Queue()
        self._down_queue = asyncio.Queue()

        logger.debug(f"Creating {self} pipelines")
        for processors in args:
            if not isinstance(processors, list):
                raise TypeError(f"SyncParallelPipeline argument {processors} is not a list")

            # We add a source at the beginning of the pipeline and a sink at the end.
            up_queue = asyncio.Queue()
            down_queue = asyncio.Queue()
            source = SyncParallelPipelineSource(up_queue)
            sink = SyncParallelPipelineSink(down_queue)

            # Keep track of sources and sinks. We also keep the output queue of
            # the source and the sinks so we can use it later.
            self._sources.append({"processor": source, "queue": down_queue})
            self._sinks.append({"processor": sink, "queue": up_queue})

            # Create pipeline
            pipeline = Pipeline(processors, source=source, sink=sink)
            self._pipelines.append(pipeline)

        logger.debug(f"Finished creating {self} pipelines")

    #
    # Frame processor
    #

    @property
    def processors(self):
        """Return the list of sub-processors contained within this processor.

        Only compound processors (e.g. pipelines and parallel pipelines) have
        sub-processors. Non-compound processors will return an empty list.

        Returns:
            The list of sub-processors if this is a compound processor.
        """
        return self._pipelines

    @property
    def entry_processors(self) -> List["FrameProcessor"]:
        """Return the list of entry processors for this processor.

        Entry processors are the first processors in a compound processor
        (e.g. pipelines, parallel pipelines). Note that pipelines can also be an
        entry processor as pipelines are processors themselves. Non-compound
        processors will simply return an empty list.

        Returns:
            The list of entry processors.
        """
        return self._sources

    def processors_with_metrics(self) -> List[FrameProcessor]:
        """Collect processors that can generate metrics from all parallel pipelines.

        Returns:
            List of frame processors that support metrics collection from all parallel paths.
        """
        return list(chain.from_iterable(p.processors_with_metrics() for p in self._pipelines))

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the parallel pipeline and all contained processors.

        Args:
            setup: Configuration for frame processor setup.
        """
        await super().setup(setup)
        await asyncio.gather(*[p.setup(setup) for p in self._pipelines])

    async def cleanup(self):
        """Clean up the parallel pipeline and all contained processors."""
        await super().cleanup()
        await asyncio.gather(*[p.cleanup() for p in self._pipelines])

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames through all parallel pipelines with synchronization.

        Distributes frames to all parallel pipelines and synchronizes their output
        to maintain proper ordering and prevent duplicate processing. Uses SyncFrame
        control frames to coordinate between parallel paths.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
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
