#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from functools import wraps
from typing import Awaitable, Callable, Union

from pipecat.frames.frames import (
    BotSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UserIdleProcessor(FrameProcessor):
    """This class is useful to check if the user is interacting with the bot within a given timeout.

    If the timeout is reached before any interaction occurred the provided callback will be called.

    The callback can be either:
    - async def callback(processor: UserIdleProcessor) -> None  # Legacy
    - async def callback(processor: UserIdleProcessor, retry_count: int) -> bool  # New

    The new style callback receives the current retry count and should return True
    to continue monitoring or False to stop.
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

    def _wrap_callback(
        self,
        callback: Union[
            Callable[["UserIdleProcessor"], Awaitable[None]],
            Callable[["UserIdleProcessor", int], Awaitable[bool]],
        ],
    ) -> Callable[["UserIdleProcessor", int], Awaitable[bool]]:
        @wraps(callback)
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

    async def _stop(self):
        self._idle_task.cancel()
        await self._idle_task

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Check for end frames before processing
        if isinstance(frame, StartFrame):
            self._create_idle_task()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()

        await self.push_frame(frame, direction)

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
        await self._stop()

    def _create_idle_task(self):
        self._idle_event = asyncio.Event()
        self._idle_task = self.get_event_loop().create_task(self._idle_task_handler())

    async def _idle_task_handler(self):
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
