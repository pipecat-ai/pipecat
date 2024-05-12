#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from typing import Callable, Coroutine, List

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class PipelineSource(FrameProcessor):

    def __init__(self, upstream_push_frame: Callable[[Frame, FrameDirection], Coroutine]):
        super().__init__()
        self._upstream_push_frame = upstream_push_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
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
        match direction:
            case FrameDirection.UPSTREAM:
                await self.push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                await self._downstream_push_frame(frame, direction)


class Pipeline(FrameProcessor):

    def __init__(self, processors: List[FrameProcessor]):
        super().__init__()

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = PipelineSource(self.push_frame)
        self._sink = PipelineSink(self.push_frame)
        self._processors: List[FrameProcessor] = [self._source] + processors + [self._sink]

        self._link_processors()

    #
    # Frame processor
    #

    async def cleanup(self):
        await self._cleanup_processors()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if direction == FrameDirection.DOWNSTREAM:
            await self._source.process_frame(frame, FrameDirection.DOWNSTREAM)
        elif direction == FrameDirection.UPSTREAM:
            await self._sink.process_frame(frame, FrameDirection.UPSTREAM)

    async def _cleanup_processors(self):
        await asyncio.gather(*[p.cleanup() for p in self._processors])

    def _link_processors(self):
        prev = self._processors[0]
        for curr in self._processors[1:]:
            prev.link(curr)
            prev = curr
