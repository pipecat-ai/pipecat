#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Producer processor for frame filtering and distribution."""

import asyncio
from typing import Awaitable, Callable, List

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


async def identity_transformer(frame: Frame):
    """Default transformer that returns the frame unchanged.

    Args:
        frame: The frame to transform.

    Returns:
        The same frame without modifications.
    """
    return frame


class ProducerProcessor(FrameProcessor):
    """A processor that filters frames and distributes them to multiple consumers.

    This processor receives frames, applies a filter to determine which frames
    should be sent to consumers (ConsumerProcessor), optionally transforms those
    frames, and distributes them to registered consumer queues. It can also pass
    frames through to the next processor in the pipeline.
    """

    def __init__(
        self,
        *,
        filter: Callable[[Frame], Awaitable[bool]],
        transformer: Callable[[Frame], Awaitable[Frame]] = identity_transformer,
        passthrough: bool = True,
    ):
        """Initialize the producer processor.

        Args:
            filter: Async function that determines if a frame should be produced.
                   Must return True for frames to be sent to consumers.
            transformer: Async function to transform frames before sending to consumers.
                        Defaults to identity_transformer which returns frames unchanged.
            passthrough: Whether to pass frames through to the next processor.
                        If True, all frames continue downstream regardless of filter result.
        """
        super().__init__()
        self._filter = filter
        self._transformer = transformer
        self._passthrough = passthrough
        self._consumers: List[asyncio.Queue] = []

    def add_consumer(self):
        """Add a new consumer and return its associated queue.

        Returns:
            asyncio.Queue: The queue for the newly added consumer.
        """
        queue = asyncio.Queue()
        self._consumers.append(queue)
        return queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame and determine whether to produce it.

        If the frame meets the filter criteria, it will be transformed and added
        to all consumer queues. If passthrough is enabled, the original frame
        will also be sent downstream.

        Args:
            frame: The frame to process.
            direction: The direction of the frame flow.
        """
        await super().process_frame(frame, direction)

        if await self._filter(frame):
            await self._produce(frame)
            if self._passthrough:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _produce(self, frame: Frame):
        """Produce a frame to all consumers."""
        for consumer in self._consumers:
            new_frame = await self._transformer(frame)
            await consumer.put(new_frame)
