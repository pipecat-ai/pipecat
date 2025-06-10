#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Awaitable, Callable, List, Optional

from pipecat.frames.frames import Frame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class IdleFrameProcessor(FrameProcessor):
    """This class waits to receive any frame or list of desired frames within a
    given timeout. If the timeout is reached before receiving any of those
    frames the provided callback will be called.
    """

    def __init__(
        self,
        *,
        callback: Callable[["IdleFrameProcessor"], Awaitable[None]],
        timeout: float,
        types: Optional[List[type]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._callback = callback
        self._timeout = timeout
        self._types = types or []
        self._idle_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._create_idle_task()

        await self.push_frame(frame, direction)

        # If we are not waiting for any specific frame set the event, otherwise
        # check if we have received one of the desired frames.
        if not self._types:
            self._idle_event.set()
        else:
            for t in self._types:
                if isinstance(frame, t):
                    self._idle_event.set()

    async def cleanup(self):
        if self._idle_task:
            await self.cancel_task(self._idle_task)

    def _create_idle_task(self):
        if not self._idle_task:
            self._idle_event = asyncio.Event()
            self._idle_task = self.create_task(self._idle_task_handler())

    async def _idle_task_handler(self):
        while True:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._timeout)
            except asyncio.TimeoutError:
                await self._callback(self)
            finally:
                self._idle_event.clear()
