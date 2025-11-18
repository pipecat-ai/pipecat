#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Idle frame processor for timeout-based callback execution."""

import asyncio
from typing import Awaitable, Callable, List, Optional

from pipecat.frames.frames import Frame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class IdleFrameProcessor(FrameProcessor):
    """Monitors frame activity and triggers callbacks on timeout.

    This processor waits to receive any frame or specific frame types within a
    given timeout period. If the timeout is reached before receiving the expected
    frames, the provided callback will be executed.
    """

    def __init__(
        self,
        *,
        callback: Callable[["IdleFrameProcessor"], Awaitable[None]],
        timeout: float,
        types: Optional[List[type]] = None,
        **kwargs,
    ):
        """Initialize the idle frame processor.

        Args:
            callback: Async callback function to execute on timeout. Receives
                this processor instance as an argument.
            timeout: Timeout duration in seconds before triggering the callback.
            types: Optional list of frame types to monitor. If None, monitors
                all frames.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        self._callback = callback
        self._timeout = timeout
        self._types = types or []
        self._idle_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and manage idle timeout monitoring.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
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
        """Clean up resources and cancel pending tasks."""
        if self._idle_task:
            await self.cancel_task(self._idle_task)

    def _create_idle_task(self):
        """Create and start the idle monitoring task."""
        if not self._idle_task:
            self._idle_event = asyncio.Event()
            self._idle_task = self.create_task(self._idle_task_handler())

    async def _idle_task_handler(self):
        """Handle idle timeout monitoring and callback execution."""
        while True:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._timeout)
            except asyncio.TimeoutError:
                await self._callback(self)
            finally:
                self._idle_event.clear()
