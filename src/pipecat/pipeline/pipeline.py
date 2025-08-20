#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline implementation for connecting and managing frame processors.

This module provides the main Pipeline class that connects frame processors
in sequence and manages frame flow between them, along with helper classes
for pipeline source and sink operations.
"""

from typing import Callable, Coroutine, List, Optional

from pipecat.frames.frames import Frame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup


class PipelineSource(FrameProcessor):
    """Source processor that forwards frames to an upstream handler.

    This processor acts as the entry point for a pipeline, forwarding
    downstream frames to the next processor and upstream frames to a
    provided upstream handler function.
    """

    def __init__(self, upstream_push_frame: Callable[[Frame, FrameDirection], Coroutine], **kwargs):
        """Initialize the pipeline source.

        Args:
            upstream_push_frame: Coroutine function to handle upstream frames.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(enable_direct_mode=True, **kwargs)
        self._upstream_push_frame = upstream_push_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and route them based on direction.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self._upstream_push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)


class PipelineSink(FrameProcessor):
    """Sink processor that forwards frames to a downstream handler.

    This processor acts as the exit point for a pipeline, forwarding
    upstream frames to the previous processor and downstream frames to a
    provided downstream handler function.
    """

    def __init__(
        self, downstream_push_frame: Callable[[Frame, FrameDirection], Coroutine], **kwargs
    ):
        """Initialize the pipeline sink.

        Args:
            downstream_push_frame: Coroutine function to handle downstream frames.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(enable_direct_mode=True, **kwargs)
        self._downstream_push_frame = downstream_push_frame

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
                await self._downstream_push_frame(frame, direction)


class Pipeline(BasePipeline):
    """Main pipeline implementation that connects frame processors in sequence.

    Creates a linear chain of frame processors with automatic source and sink
    processors for external frame handling. Manages processor lifecycle and
    provides metrics collection from contained processors.
    """

    def __init__(
        self,
        processors: List[FrameProcessor],
        *,
        source: Optional[FrameProcessor] = None,
        sink: Optional[FrameProcessor] = None,
    ):
        """Initialize the pipeline with a list of processors.

        Args:
            processors: List of frame processors to connect in sequence.
            source: An optional pipeline source processor.
            sink: An optional pipeline sink processor.
        """
        super().__init__(enable_direct_mode=True)

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = source or PipelineSource(self.push_frame, name=f"{self}::Source")
        self._sink = sink or PipelineSink(self.push_frame, name=f"{self}::Sink")
        self._processors: List[FrameProcessor] = [self._source] + processors + [self._sink]

        self._link_processors()

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
        return self._processors

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
        return [self._source]

    def processors_with_metrics(self):
        """Return processors that can generate metrics.

        Recursively collects all processors that support metrics generation,
        including those from nested pipelines.

        Returns:
            List of frame processors that can generate metrics.
        """
        services = []
        for p in self.processors:
            if p.can_generate_metrics():
                services.append(p)
            services.extend(p.processors_with_metrics())
        return services

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the pipeline and all contained processors.

        Args:
            setup: Configuration for frame processor setup.
        """
        await super().setup(setup)
        await self._setup_processors(setup)

    async def cleanup(self):
        """Clean up the pipeline and all contained processors."""
        await super().cleanup()
        await self._cleanup_processors()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames by routing them through the pipeline.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            await self._source.queue_frame(frame, FrameDirection.DOWNSTREAM)
        elif direction == FrameDirection.UPSTREAM:
            await self._sink.queue_frame(frame, FrameDirection.UPSTREAM)

    async def _setup_processors(self, setup: FrameProcessorSetup):
        """Set up all processors in the pipeline."""
        for p in self._processors:
            await p.setup(setup)

    async def _cleanup_processors(self):
        """Clean up all processors in the pipeline."""
        for p in self._processors:
            await p.cleanup()

    def _link_processors(self):
        """Link all processors in sequence and set their parent."""
        prev = self._processors[0]
        for curr in self._processors[1:]:
            prev.link(curr)
            prev = curr
