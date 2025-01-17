#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from functools import wraps
from typing import Awaitable, Callable, Union, cast

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
    or BotSpeaking). Supports both legacy and new-style callbacks for handling idle events.

    Args:
        callback: Function to call when user is idle. Can be either:
            - Legacy: async def(processor) -> None
            - New: async def(processor, retry_count: int) -> bool
        timeout: Seconds to wait before considering user idle
        **kwargs: Additional arguments passed to FrameProcessor

    Example:
    async def handle_idle(processor, retry_count: int) -> bool:
        if retry_count <= 3:
            await send_reminder()
            return True
        return False

    processor = UserIdleProcessor(
        callback=handle_idle,
        timeout=5.0
    )
    """

    def __init__(
        self,
        *,
        callback: Union[
            Callable[["UserIdleProcessor"], Awaitable[None]],  # Old signature
            Callable[["UserIdleProcessor", int], Awaitable[bool]],  # New signature
        ],
        timeout: float,
        **kwargs,
    ):
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
        """Wraps callback to support both old and new-style signatures.

        Returns:
            Wrapped callback that returns bool to indicate whether to continue monitoring
        """
        # Cast the callback to the new-style signature for wraps
        wrapped_cb = cast(Callable[["UserIdleProcessor", int], Awaitable[bool]], callback)

        @wraps(wrapped_cb)
        async def wrapper(processor: "UserIdleProcessor", retry_count: int) -> bool:
            # Check callback signature
            import inspect

            sig = inspect.signature(callback)
            param_count = len(sig.parameters)

            if param_count == 1:
                # Old callback
                await callback(processor)  # type: ignore
                return True  # Always continue for backwards compatibility
            else:
                # New callback
                return await callback(processor, retry_count)  # type: ignore

        return wrapper

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
                self._retry_count = 0
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

        Runs in a loop until cancelled or callback indicates completion.
        """
        while True:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._timeout)
            except asyncio.TimeoutError:
                if not self._interrupted:
                    self._retry_count += 1
                    should_continue = await self._callback(self, self._retry_count)
                    if not should_continue:
                        await self._stop()
                        break
            except asyncio.CancelledError:
                break
            finally:
                self._idle_event.clear()
