#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Awaitable, Callable, List

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio.watchdog_queue import WatchdogQueue


async def identity_transformer(frame: Frame):
    return frame


class ProducerProcessor(FrameProcessor):
    """This class optionally passes-through received frames and decides if those
    frames should be sent to consumers based on a user-defined filter. The
    frames can be transformed into a different type of frame before being
    sending them to the consumers. More than one consumer can be added.

    """

    def __init__(
        self,
        *,
        filter: Callable[[Frame], Awaitable[bool]],
        transformer: Callable[[Frame], Awaitable[Frame]] = identity_transformer,
        passthrough: bool = True,
    ):
        super().__init__()
        self._filter = filter
        self._transformer = transformer
        self._passthrough = passthrough
        self._consumers: List[asyncio.Queue] = []

    def add_consumer(self):
        """
        Adds a new consumer and returns its associated queue.

        Returns:
            asyncio.Queue: The queue for the newly added consumer.
        """
        queue = WatchdogQueue(self.task_manager)
        self._consumers.append(queue)
        return queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Processes an incoming frame and determines whether to produce it as a ProducerItem.

        If the frame meets the produce criteria, it will be added to the consumer queues.
        If passthrough is enabled, the frame will also be sent to consumers.

        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction of the frame.
        """
        await super().process_frame(frame, direction)

        if await self._filter(frame):
            await self._produce(frame)
            if self._passthrough:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _produce(self, frame: Frame):
        for consumer in self._consumers:
            new_frame = await self._transformer(frame)
            await consumer.put(new_frame)
