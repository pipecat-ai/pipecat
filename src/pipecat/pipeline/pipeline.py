#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Callable, Coroutine, List

from pipecat.frames.frames import Frame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup


class PipelineSource(FrameProcessor):
    def __init__(self, upstream_push_frame: Callable[[Frame, FrameDirection], Coroutine]):
        super().__init__()
        self._upstream_push_frame = upstream_push_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self._upstream_push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)


class PipelineSink(FrameProcessor):
    def __init__(self, downstream_push_frame: Callable[[Frame, FrameDirection], Coroutine]):
        super().__init__()
        self._downstream_push_frame = downstream_push_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self.push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                await self._downstream_push_frame(frame, direction)


class Pipeline(BasePipeline):
    def __init__(self, processors: List[FrameProcessor]):
        super().__init__()

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = PipelineSource(self.push_frame)
        self._sink = PipelineSink(self.push_frame)
        self._processors: List[FrameProcessor] = [self._source] + processors + [self._sink]

        self._link_processors()

    #
    # BasePipeline
    #

    def processors_with_metrics(self):
        services = []
        for p in self._processors:
            if isinstance(p, BasePipeline):
                services.extend(p.processors_with_metrics())
            elif p.can_generate_metrics():
                services.append(p)
        return services

    #
    # Frame processor
    #

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._setup_processors(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._cleanup_processors()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            await self._source.queue_frame(frame, FrameDirection.DOWNSTREAM)
        elif direction == FrameDirection.UPSTREAM:
            await self._sink.queue_frame(frame, FrameDirection.UPSTREAM)

    async def _setup_processors(self, setup: FrameProcessorSetup):
        for p in self._processors:
            await p.setup(setup)

    async def _cleanup_processors(self):
        for p in self._processors:
            await p.cleanup()

    def _link_processors(self):
        prev = self._processors[0]
        for curr in self._processors[1:]:
            prev.set_parent(self)
            prev.link(curr)
            prev = curr
        prev.set_parent(self)
