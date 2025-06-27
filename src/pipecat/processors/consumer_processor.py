#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Awaitable, Callable, Optional

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.producer_processor import ProducerProcessor, identity_transformer
from pipecat.utils.asyncio.watchdog_queue import WatchdogQueue


class ConsumerProcessor(FrameProcessor):
    """This class passes-through frames and also consumes frames from a
    producer's queue. When a frame from a producer queue is received it will be
    pushed to the specified direction. The frames can be transformed into a
    different type of frame before being pushed.

    """

    def __init__(
        self,
        *,
        producer: ProducerProcessor,
        transformer: Callable[[Frame], Awaitable[Frame]] = identity_transformer,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._transformer = transformer
        self._direction = direction
        self._producer = producer
        self._consumer_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            await self._stop(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)

        await self.push_frame(frame, direction)

    async def _start(self, _: StartFrame):
        if not self._consumer_task:
            self._queue: WatchdogQueue = self._producer.add_consumer()
            self._consumer_task = self.create_task(self._consumer_task_handler())

    async def _stop(self, _: EndFrame):
        if self._consumer_task:
            await self.cancel_task(self._consumer_task)

    async def _cancel(self, _: CancelFrame):
        if self._consumer_task:
            await self.cancel_task(self._consumer_task)

    async def _consumer_task_handler(self):
        while True:
            frame = await self._queue.get()
            new_frame = await self._transformer(frame)
            await self.push_frame(new_frame, self._direction)
