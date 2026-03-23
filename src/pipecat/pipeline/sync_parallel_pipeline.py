#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Synchronized parallel pipeline that holds output until all branches finish.

A SyncParallelPipeline fans each inbound frame out to multiple parallel pipelines
and waits for every pipeline to finish processing before releasing any of the
resulting output frames. This ensures that all frames produced in response to a
single input frame are emitted together.

System frames (except EndFrame) are exempt from this synchronization — they pass
straight through without waiting, since they are expected to race ahead of
regular data frames.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import List

from loguru import logger

from pipecat.frames.frames import ControlFrame, EndFrame, Frame, SystemFrame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup


class FrameOrder(Enum):
    """Controls the order in which synchronized frames are pushed downstream.

    When multiple parallel pipelines produce output for the same input frame,
    this setting determines the order in which those output frames are pushed.

    Attributes:
        ARRIVAL: Frames are pushed in the order they arrive from any pipeline.
            This is the default and matches the behavior of prior versions.
        PIPELINE: Frames are pushed in pipeline definition order — all frames
            from the first pipeline are pushed, then all frames from the second
            pipeline, and so on. Useful when the relative ordering between
            pipelines matters (e.g. ensuring image frames precede audio frames).
    """

    ARRIVAL = "arrival"
    PIPELINE = "pipeline"


@dataclass
class SyncFrame(ControlFrame):
    """Sentinel frame used to detect when a parallel pipeline has finished processing.

    After sending a real frame into a parallel pipeline, a SyncFrame is sent
    behind it. When the SyncFrame emerges from the pipeline's output, we know
    all output frames for the preceding input have been produced.
    """

    pass


class SyncParallelPipelineSource(FrameProcessor):
    """Bookend processor placed at the start of each parallel pipeline.

    Forwards downstream frames into the pipeline and captures upstream frames
    into a queue so the parent SyncParallelPipeline can release them later.
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
    """Bookend processor placed at the end of each parallel pipeline.

    Captures downstream output frames into a queue so the parent
    SyncParallelPipeline can release them later, and forwards upstream
    frames back through the pipeline.
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
    """Fans each input frame to parallel pipelines then holds output until every pipeline finishes.

    For each inbound frame the pipeline:

    1. Sends the frame into every parallel pipeline.
    2. Sends a ``SyncFrame`` sentinel behind it in each pipeline.
    3. Waits until every pipeline has produced its ``SyncFrame``, meaning all
       output for that input is ready.
    4. Releases the collected output frames (deduplicating by frame id, since
       the same frame may emerge from more than one branch).

    System frames (except ``EndFrame``) bypass this mechanism entirely — they are
    forwarded through each pipeline and pushed immediately, since system frames
    are expected to race ahead of regular data frames.

    By default, output frames are pushed in the order they arrive from any pipeline
    (``FrameOrder.ARRIVAL``). Set ``frame_order=FrameOrder.PIPELINE`` to push frames
    in pipeline definition order instead — all output from the first pipeline, then
    the second, and so on.
    """

    def __init__(self, *args, frame_order: FrameOrder = FrameOrder.ARRIVAL):
        """Initialize the synchronous parallel pipeline.

        Args:
            *args: Variable number of processor lists, each representing a parallel
                pipeline path. Each argument should be a list of FrameProcessor instances.
            frame_order: Controls the order in which synchronized output frames are
                pushed. ``FrameOrder.ARRIVAL`` (default) pushes frames in the order they arrive.
                ``FrameOrder.PIPELINE`` pushes all frames from the first pipeline
                before the second, and so on.

        Raises:
            Exception: If no arguments are provided.
            TypeError: If any argument is not a list of processors.
        """
        super().__init__()
        self._frame_order = frame_order

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
        return [s["processor"] for s in self._sources]

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
        """Send a frame through all parallel pipelines and release output once all finish.

        System frames (except EndFrame) skip synchronization and pass straight
        through. All other frames are fanned out to every pipeline, and output is
        held until every pipeline signals completion (via SyncFrame).

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        # SystemFrames are simply passed through all internal pipelines without
        # draining queued output. This avoids the race condition where a
        # SystemFrame's wait_for_sync steals frames from a concurrent
        # non-SystemFrame's wait_for_sync.
        if isinstance(frame, SystemFrame):
            if direction == FrameDirection.UPSTREAM:
                for s in self._sinks:
                    await s["processor"].process_frame(frame, direction)
            elif direction == FrameDirection.DOWNSTREAM:
                for s in self._sources:
                    await s["processor"].process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        use_pipeline_order = self._frame_order == FrameOrder.PIPELINE

        # The last processor of each pipeline needs to be synchronous otherwise
        # this element won't work. Since we know it should be synchronous we
        # push a SyncFrame. Since frames are ordered we know this frame will be
        # pushed after the synchronous processor has pushed its data allowing us
        # to synchronize all the internal pipelines by waiting for the
        # SyncFrame in all of them.
        #
        # In ARRIVAL mode, output frames are put onto a shared main_queue as
        # they arrive. In PIPELINE mode, they are accumulated in a per-pipeline
        # list and returned so the caller can drain them in definition order.
        async def wait_for_sync(
            obj, main_queue: asyncio.Queue, frame: Frame, direction: FrameDirection
        ) -> list[Frame]:
            processor = obj["processor"]
            queue = obj["queue"]
            output_frames: list[Frame] = []

            await processor.process_frame(frame, direction)

            if isinstance(frame, EndFrame):
                new_frame = await queue.get()
                if isinstance(new_frame, EndFrame):
                    if use_pipeline_order:
                        output_frames.append(new_frame)
                    else:
                        await main_queue.put(new_frame)
                else:
                    while not isinstance(new_frame, EndFrame):
                        if use_pipeline_order:
                            output_frames.append(new_frame)
                        else:
                            await main_queue.put(new_frame)
                        queue.task_done()
                        new_frame = await queue.get()
            else:
                await processor.process_frame(SyncFrame(), direction)
                new_frame = await queue.get()
                while not isinstance(new_frame, SyncFrame):
                    if use_pipeline_order:
                        output_frames.append(new_frame)
                    else:
                        await main_queue.put(new_frame)
                    queue.task_done()
                    new_frame = await queue.get()

            return output_frames

        if direction == FrameDirection.UPSTREAM:
            # If we get an upstream frame we process it in each sink.
            frames_per_pipeline = await asyncio.gather(
                *[wait_for_sync(s, self._up_queue, frame, direction) for s in self._sinks]
            )
        elif direction == FrameDirection.DOWNSTREAM:
            # If we get a downstream frame we process it in each source.
            frames_per_pipeline = await asyncio.gather(
                *[wait_for_sync(s, self._down_queue, frame, direction) for s in self._sources]
            )

        if use_pipeline_order:
            # Push frames in pipeline definition order, deduplicating by id.
            seen_ids = set()
            for pipeline_frames in frames_per_pipeline:
                for f in pipeline_frames:
                    if f.id not in seen_ids:
                        await self.push_frame(f, direction)
                        seen_ids.add(f.id)
        else:
            # ARRIVAL mode: drain the shared queues in the order frames arrived.
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
