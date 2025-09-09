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

from itertools import chain
from typing import Dict, List

from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline, PipelineSink, PipelineSource
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup


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
        # We don't set it to direct mode because we use frame pausing and that
        # requires queues.
        super().__init__()

        if len(args) == 0:
            raise Exception(f"ParallelPipeline needs at least one argument")

        self._pipelines = []

        self._seen_ids = set()
        self._frame_counter: Dict[int, int] = {}

        logger.debug(f"Creating {self} pipelines")

        for processors in args:
            if not isinstance(processors, list):
                raise TypeError(f"ParallelPipeline argument {processors} is not a list")

            num_pipelines = len(self._pipelines)

            # We add a source before the pipeline and a sink after so we control
            # the frames that are pushed upstream and downstream.
            source = PipelineSource(
                self._parallel_push_frame, name=f"{self}::Source{num_pipelines}"
            )
            sink = PipelineSink(self._pipeline_sink_push_frame, name=f"{self}::Sink{num_pipelines}")

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
        return self._pipelines

    def processors_with_metrics(self) -> List[FrameProcessor]:
        """Collect processors that can generate metrics from all parallel branches.

        Returns:
            List of frame processors that support metrics collection from all branches.
        """
        return list(chain.from_iterable(p.processors_with_metrics() for p in self._pipelines))

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the parallel pipeline and all its branches.

        Args:
            setup: Configuration for frame processor setup.

        Raises:
            TypeError: If any processor list argument is not actually a list.
        """
        await super().setup(setup)
        for p in self._pipelines:
            await p.setup(setup)

    async def cleanup(self):
        """Clean up the parallel pipeline and all its branches."""
        await super().cleanup()
        for p in self._pipelines:
            await p.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames through all parallel branches with lifecycle coordination.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        # Parallel pipeline synchronized frames.
        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            self._frame_counter[frame.id] = len(self._pipelines)
            await self.pause_processing_system_frames()
            await self.pause_processing_frames()

        # Process frames in each of the sub-pipelines.
        for p in self._pipelines:
            await p.queue_frame(frame, direction)

    async def _parallel_push_frame(self, frame: Frame, direction: FrameDirection):
        """Push frames while avoiding duplicates using frame ID tracking."""
        if frame.id not in self._seen_ids:
            self._seen_ids.add(frame.id)
            await self.push_frame(frame, direction)

    async def _pipeline_sink_push_frame(self, frame: Frame, direction: FrameDirection):
        # Parallel pipeline synchronized frames.
        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            # Decrement counter.
            frame_counter = self._frame_counter.get(frame.id, 0)
            if frame_counter > 0:
                self._frame_counter[frame.id] -= 1
                frame_counter = self._frame_counter[frame.id]

            # Only push the frame when all pipelines have processed it.
            if frame_counter == 0:
                await self._parallel_push_frame(frame, direction)
                await self.resume_processing_system_frames()
                await self.resume_processing_frames()
        else:
            await self._parallel_push_frame(frame, direction)
