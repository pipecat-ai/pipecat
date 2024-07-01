#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.frames.frames import EndFrame, Frame, StartInterruptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AsyncFrameProcessor(FrameProcessor):

    def __init__(
            self,
            *,
            name: str | None = None,
            loop: asyncio.AbstractEventLoop | None = None,
            **kwargs):
        super().__init__(name=name, loop=loop, **kwargs)

        self._create_push_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruptions(frame)

    async def queue_frame(
            self,
            frame: Frame,
            direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await self._push_queue.put((frame, direction))

    async def cleanup(self):
        self._push_frame_task.cancel()
        await self._push_frame_task

    async def _handle_interruptions(self, frame: Frame):
        # Cancel the task. This will stop pushing frames downstream.
        self._push_frame_task.cancel()
        await self._push_frame_task
        # Push an out-of-band frame (i.e. not using the ordered push
        # frame task).
        await self.push_frame(frame)
        # Create a new queue and task.
        self._create_push_task()

    def _create_push_task(self):
        self._push_queue = asyncio.Queue()
        self._push_frame_task = self.get_event_loop().create_task(self._push_frame_task_handler())

    async def _push_frame_task_handler(self):
        running = True
        while running:
            try:
                (frame, direction) = await self._push_queue.get()
                await self.push_frame(frame, direction)
                running = not isinstance(frame, EndFrame)
            except asyncio.CancelledError:
                break
