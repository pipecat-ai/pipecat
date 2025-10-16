#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User idle detection and timeout handling for Pipecat."""

import asyncio
import inspect
from typing import Awaitable, Callable, Union

from pipecat.frames.frames import (
    BotSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UserIdleProcessor(FrameProcessor):
    """Monitors user inactivity and triggers callbacks after timeout periods.

    This processor tracks user activity and triggers configurable callbacks when
    users become idle. It starts monitoring only after the first conversation
    activity and supports both basic and retry-based callback patterns.

    Example::

        # Retry callback:
        async def handle_idle(processor: "UserIdleProcessor", retry_count: int) -> bool:
            if retry_count < 3:
                await send_reminder("Are you still there?")
                return True
            return False

        # Basic callback:
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
        callback: Union[
            Callable[["UserIdleProcessor"], Awaitable[None]],  # Basic
            Callable[["UserIdleProcessor", int], Awaitable[bool]],  # Retry
        ],
        timeout: float,
        **kwargs,
    ):
        """Initialize the user idle processor.

        Args:
            callback: Function to call when user is idle. Can be either a basic
                callback taking only the processor, or a retry callback taking
                the processor and retry count. Retry callbacks should return
                True to continue monitoring or False to stop.
            timeout: Seconds to wait before considering user idle.
            **kwargs: Additional arguments passed to FrameProcessor.
        """
        super().__init__(**kwargs)
        self._callback = self._wrap_callback(callback)
        self._timeout = timeout
        self._retry_count = 0
        self._interrupted = False
        self._conversation_started = False
        self._idle_task = None
        self._idle_event = asyncio.Event()

    def _wrap_callback(
        self,
        callback: Union[
            Callable[["UserIdleProcessor"], Awaitable[None]],
            Callable[["UserIdleProcessor", int], Awaitable[bool]],
        ],
    ) -> Callable[["UserIdleProcessor", int], Awaitable[bool]]:
        """Wraps callback to support both basic and retry signatures.

        Args:
            callback: The callback function to wrap.

        Returns:
            A wrapped callback that returns bool to indicate whether to continue monitoring.
        """
        sig = inspect.signature(callback)
        param_count = len(sig.parameters)

        async def wrapper(processor: "UserIdleProcessor", retry_count: int) -> bool:
            if param_count == 1:
                # Basic callback
                await callback(processor)  # type: ignore
                return True
            else:
                # Retry callback
                return await callback(processor, retry_count)  # type: ignore

        return wrapper

    def _create_idle_task(self) -> None:
        """Creates the idle task if it hasn't been created yet."""
        if not self._idle_task:
            self._idle_task = self.create_task(self._idle_task_handler())

    @property
    def retry_count(self) -> int:
        """Get the current retry count.

        Returns:
            The number of times the idle callback has been triggered.
        """
        return self._retry_count

    async def _stop(self) -> None:
        """Stops and cleans up the idle monitoring task."""
        if self._idle_task:
            await self.cancel_task(self._idle_task)
            self._idle_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes incoming frames and manages idle monitoring state.

        Args:
            frame: The frame to process.
            direction: Direction of the frame flow.
        """
        await super().process_frame(frame, direction)

        # Check for end frames before processing
        if isinstance(frame, (EndFrame, CancelFrame)):
            # Stop the idle task, if it exists
            await self._stop()
            # Push the frame down the pipeline
            await self.push_frame(frame, direction)
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
                self._retry_count = 0  # Reset retry count when user speaks
                self._interrupted = True
                self._idle_event.set()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._interrupted = False
                self._idle_event.set()
            elif isinstance(frame, BotSpeakingFrame):
                self._idle_event.set()
            elif isinstance(frame, FunctionCallInProgressFrame):
                # Function calls can take longer than the timeout, so we want to prevent idle callbacks
                self._interrupted = True
                self._idle_event.set()
            elif isinstance(frame, FunctionCallResultFrame):
                self._interrupted = False
                self._idle_event.set()

    async def cleanup(self) -> None:
        """Cleans up resources when processor is shutting down."""
        await super().cleanup()
        if self._idle_task:  # Only stop if task exists
            await self._stop()

    async def _idle_task_handler(self) -> None:
        """Monitors for idle timeout and triggers callbacks.

        Runs in a loop until cancelled or callback indicates completion.
        """
        running = True
        while running:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._timeout)
            except asyncio.TimeoutError:
                if not self._interrupted:
                    self._retry_count += 1
                    running = await self._callback(self, self._retry_count)
            finally:
                self._idle_event.clear()
