#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Awaitable, Callable

from pipecat.frames.frames import (
    BotSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UserIdleProcessor(FrameProcessor):
    """Monitors user inactivity and triggers callbacks after timeout periods.

    Starts monitoring only after the first conversation activity (UserStartedSpeaking
    or BotSpeaking).

    Args:
        callback: Function to call when user is idle
        timeout: Seconds to wait before considering user idle
        **kwargs: Additional arguments passed to FrameProcessor

    Example:
        async def handle_idle(processor: "UserIdleProcessor") -> None:
            await send_reminder("Are you still there?")

        processor = UserIdleProcessor(
            callback=handle_idle,
            timeout=5.0
        )
    """

    def __init__(
        self,
        *,
        callback: Callable[["UserIdleProcessor"], Awaitable[None]],
        timeout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._callback = callback
        self._timeout = timeout
        self._interrupted = False
        self._conversation_started = False
        self._idle_task = None
        self._idle_event = asyncio.Event()

    def _create_idle_task(self):
        """Create the idle task if it hasn't been created yet."""
        if self._idle_task is None:
            self._idle_task = self.get_event_loop().create_task(self._idle_task_handler())

    async def _stop(self):
        """Stops and cleans up the idle monitoring task."""
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass  # Expected when task is cancelled
            self._idle_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes incoming frames and manages idle monitoring state.

        Args:
            frame: The frame to process
            direction: Direction of the frame flow
        """
        await super().process_frame(frame, direction)

        # Check for end frames before processing
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self.push_frame(frame, direction)  # Push the frame down the pipeline
            if self._idle_task:
                await self._stop()  # Stop the idle task, if it exists
            return

        await self.push_frame(frame, direction)

        # Start monitoring on first conversation activity
        if not self._conversation_started and isinstance(
            frame, (UserStartedSpeakingFrame, BotSpeakingFrame)
        ):
            self._conversation_started = True
            self._create_idle_task()

        # Only process these events if conversation has started
        if self._conversation_started:
            # We shouldn't call the idle callback if the user or the bot are speaking
            if isinstance(frame, UserStartedSpeakingFrame):
                self._interrupted = True
                self._idle_event.set()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._interrupted = False
                self._idle_event.set()
            elif isinstance(frame, BotSpeakingFrame):
                self._idle_event.set()

    async def cleanup(self):
        """Cleans up resources when processor is shutting down."""
        if self._idle_task:  # Only stop if task exists
            await self._stop()

    async def _idle_task_handler(self):
        """Monitors for idle timeout and triggers callbacks.

        Runs in a loop until cancelled.
        """
        while True:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._timeout)
            except asyncio.TimeoutError:
                if not self._interrupted:
                    await self._callback(self)
            except asyncio.CancelledError:
                break
            finally:
                self._idle_event.clear()
