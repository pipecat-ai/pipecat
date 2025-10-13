#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Consumer processor for consuming frames from ProducerProcessor queues."""

import asyncio
from typing import Awaitable, Callable, Optional

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.producer_processor import ProducerProcessor, identity_transformer


class ConsumerProcessor(FrameProcessor):
    """Frame processor that consumes frames from a ProducerProcessor's queue.

    This processor passes through frames normally while also consuming frames
    from a ProducerProcessor's queue. When frames are received from the producer
    queue, they are optionally transformed and pushed in the specified direction.
    """

    def __init__(
        self,
        *,
        producer: ProducerProcessor,
        transformer: Callable[[Frame], Awaitable[Frame]] = identity_transformer,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        **kwargs,
    ):
        """Initialize the consumer processor.

        Args:
            producer: The producer processor to consume frames from.
            transformer: Function to transform frames before pushing. Defaults to identity_transformer.
            direction: Direction to push consumed frames. Defaults to DOWNSTREAM.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._transformer = transformer
        self._direction = direction
        self._producer = producer
        self._consumer_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle lifecycle events.

        Args:
            frame: The frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            await self._stop(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)

        await self.push_frame(frame, direction)

    async def _start(self, _: StartFrame):
        """Start the consumer task and register with the producer."""
        if not self._consumer_task:
            self._queue = self._producer.add_consumer()
            self._consumer_task = self.create_task(self._consumer_task_handler())

    async def _stop(self, _: EndFrame):
        """Stop the consumer task."""
        if self._consumer_task:
            await self.cancel_task(self._consumer_task)

    async def _cancel(self, _: CancelFrame):
        """Cancel the consumer task."""
        if self._consumer_task:
            await self.cancel_task(self._consumer_task)

    async def _consumer_task_handler(self):
        """Handle consuming frames from the producer queue."""
        while True:
            frame = await self._queue.get()
            new_frame = await self._transformer(frame)
            await self.push_frame(new_frame, self._direction)
