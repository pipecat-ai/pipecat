#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Parallel pipeline implementation for concurrent frame processing.

This module provides a parallel pipeline that processes frames through multiple
sub-pipelines concurrently, with coordination for system frames and proper
handling of pipeline lifecycle events.
"""

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
    """Source processor for parallel pipeline branches.

    Handles frame routing for parallel pipeline inputs, directing system frames
    to the parent push function and other upstream frames to a queue for processing.
    """

    def __init__(
        self,
        upstream_queue: asyncio.Queue,
        push_frame_func: Callable[[Frame, FrameDirection], Awaitable[None]],
    ):
        """Initialize the parallel pipeline source.

        Args:
            upstream_queue: Queue for collecting upstream frames from this branch.
            push_frame_func: Function to push frames to the parent parallel pipeline.
        """
        super().__init__()
        self._up_queue = upstream_queue
        self._push_frame_func = push_frame_func

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with special handling for system frames.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
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
    """Sink processor for parallel pipeline branches.

    Handles frame routing for parallel pipeline outputs, directing system frames
    to the parent push function and other downstream frames to a queue for coordination.
    """

    def __init__(
        self,
        downstream_queue: asyncio.Queue,
        push_frame_func: Callable[[Frame, FrameDirection], Awaitable[None]],
    ):
        """Initialize the parallel pipeline sink.

        Args:
            downstream_queue: Queue for collecting downstream frames from this branch.
            push_frame_func: Function to push frames to the parent parallel pipeline.
        """
        super().__init__()
        self._down_queue = downstream_queue
        self._push_frame_func = push_frame_func

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with special handling for system frames.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
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
    """Pipeline that processes frames through multiple sub-pipelines concurrently.

    Creates multiple parallel processing branches from the provided processor lists,
    coordinating frame flow and ensuring proper synchronization of lifecycle events
    like EndFrames. Each branch runs independently while system frames are handled
    specially to maintain pipeline coordination.
    """

    def __init__(self, *args):
        """Initialize the parallel pipeline with processor lists.

        Args:
            *args: Variable number of processor lists, each becoming a parallel branch.

        Raises:
            Exception: If no processor lists are provided.
            TypeError: If any argument is not a list of processors.
        """
        super().__init__()

        if len(args) == 0:
            raise Exception(f"ParallelPipeline needs at least one argument")

        self._args = args
        self._sources = []
        self._sinks = []
        self._pipelines = []

        self._seen_ids = set()
        self._endframe_counter: Dict[int, int] = {}
        self._start_frame_counter: Dict[int, int] = {}
        self._started = False

        self._up_task = None
        self._down_task = None

    #
    # BasePipeline
    #

    def processors_with_metrics(self) -> List[FrameProcessor]:
        """Collect processors that can generate metrics from all parallel branches.

        Returns:
            List of frame processors that support metrics collection from all branches.
        """
        return list(chain.from_iterable(p.processors_with_metrics() for p in self._pipelines))

    #
    # Frame processor
    #

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the parallel pipeline and all its branches.

        Args:
            setup: Configuration for frame processor setup.

        Raises:
            TypeError: If any processor list argument is not actually a list.
        """
        await super().setup(setup)

        self._up_queue = WatchdogQueue(setup.task_manager)
        self._down_queue = WatchdogQueue(setup.task_manager)

        logger.debug(f"Creating {self} pipelines")
        for processors in self._args:
            if not isinstance(processors, list):
                raise TypeError(f"ParallelPipeline argument {processors} is not a list")

            # We will add a source before the pipeline and a sink after.
            source = ParallelPipelineSource(self._up_queue, self._parallel_push_frame)
            sink = ParallelPipelineSink(self._down_queue, self._pipeline_sink_push_frame)
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
        """Clean up the parallel pipeline and all its branches."""
        await super().cleanup()
        await asyncio.gather(*[s.cleanup() for s in self._sources])
        await asyncio.gather(*[p.cleanup() for p in self._pipelines])
        await asyncio.gather(*[s.cleanup() for s in self._sinks])

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames through all parallel branches with lifecycle coordination.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._start_frame_counter[frame.id] = len(self._pipelines)
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
        """Start the parallel pipeline processing tasks."""
        await self._create_tasks()

    async def _stop(self):
        """Stop all parallel pipeline processing tasks."""
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
        """Cancel all parallel pipeline processing tasks."""
        if self._up_task:
            self._up_queue.cancel()
            await self.cancel_task(self._up_task)
            self._up_task = None
        if self._down_task:
            self._down_queue.cancel()
            await self.cancel_task(self._down_task)
            self._down_task = None

    async def _create_tasks(self):
        """Create upstream and downstream processing tasks if not already running."""
        if not self._up_task:
            self._up_task = self.create_task(self._process_up_queue())
        if not self._down_task:
            self._down_task = self.create_task(self._process_down_queue())

    async def _drain_queue(self, queue: asyncio.Queue):
        try:
            while not queue.empty():
                queue.get_nowait()
        except asyncio.QueueEmpty:
            logger.debug(f"Draining {self} queue already empty")

    async def _drain_queues(self):
        """Drain all frames from upstream and downstream queues."""
        await self._drain_queue(self._up_queue)
        await self._drain_queue(self._down_queue)

    async def _handle_interruption(self):
        """Handle interruption by cancelling tasks, draining queues, and restarting."""
        await self._cancel()
        await self._drain_queues()
        await self._create_tasks()

    async def _parallel_push_frame(self, frame: Frame, direction: FrameDirection):
        """Push frames while avoiding duplicates using frame ID tracking."""
        if frame.id not in self._seen_ids:
            self._seen_ids.add(frame.id)
            await self.push_frame(frame, direction)

    async def _pipeline_sink_push_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            # Decrement counter and check if all pipelines have processed the StartFrame
            start_frame_counter = self._start_frame_counter.get(frame.id, 0)
            if start_frame_counter > 0:
                self._start_frame_counter[frame.id] -= 1
                start_frame_counter = self._start_frame_counter[frame.id]

            # Only push the StartFrame when all pipelines have processed it
            if start_frame_counter == 0:
                self._started = True
                await self._start(frame)
                await self._parallel_push_frame(frame, direction)
        else:
            if self._started:
                await self._parallel_push_frame(frame, direction)
            else:
                await self._down_queue.put(frame)

    async def _process_up_queue(self):
        """Process upstream frames from all parallel branches."""
        while True:
            frame = await self._up_queue.get()
            await self._parallel_push_frame(frame, FrameDirection.UPSTREAM)
            self._up_queue.task_done()

    async def _process_down_queue(self):
        """Process downstream frames with EndFrame coordination.

        Coordinates EndFrames to ensure they are only pushed upstream once
        all parallel branches have completed processing them.
        """
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
