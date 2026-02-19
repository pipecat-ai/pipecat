#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a controller for managing user idle detection."""

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    UserIdleTimeoutUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class UserIdleController(BaseObject):
    """Controller for managing user idle detection.

    This class monitors user activity and triggers an event when the user has been
    idle (not speaking) for a configured timeout period after the bot finishes
    speaking. The timer starts when BotStoppedSpeakingFrame is received and is
    cancelled when someone starts speaking again (UserStartedSpeakingFrame or
    BotStartedSpeakingFrame).

    The timer is suppressed while a user turn is in progress to avoid false
    triggers during interruptions (where BotStoppedSpeakingFrame arrives while
    the user is still speaking).

    Event handlers available:

    - on_user_turn_idle: Emitted when the user has been idle for the timeout period.

    Example::

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            # Handle user idle - send reminder, prompt, etc.
            ...
    """

    def __init__(
        self,
        *,
        user_idle_timeout: float = 0,
    ):
        """Initialize the user idle controller.

        Args:
            user_idle_timeout: Timeout in seconds before considering the user idle.
                0 disables idle detection.
        """
        super().__init__()

        self._user_idle_timeout = user_idle_timeout

        self._task_manager: Optional[BaseTaskManager] = None

        self._user_turn_in_progress: bool = False
        self._function_calls_in_progress: int = 0
        self._idle_timer_task: Optional[asyncio.Task] = None

        self._register_event_handler("on_user_turn_idle", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} user idle controller was not properly setup")
        return self._task_manager

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the controller with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        self._task_manager = task_manager

    async def cleanup(self):
        """Cleanup the controller."""
        await super().cleanup()
        await self._cancel_idle_timer()

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to track user activity state.

        Args:
            frame: The frame to be processed.
        """
        if isinstance(frame, UserIdleTimeoutUpdateFrame):
            self._user_idle_timeout = frame.timeout
            if self._user_idle_timeout <= 0:
                await self._cancel_idle_timer()
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            # Only start the timer if the user isn't mid-turn and no function
            # calls are pending.
            #
            # Interruption case: the frame order is UserStartedSpeaking →
            # BotStoppedSpeaking → (user keeps talking) → UserStoppedSpeaking.
            # Without the user-turn guard the timer would start while the user
            # is still speaking.
            #
            # Function call case: normally FunctionCallsStarted arrives after
            # BotStoppedSpeaking and cancels the timer directly. But a race
            # condition can cause FunctionCallsStarted to arrive before
            # BotStoppedSpeaking when pushing a TTSSpeakFrame in the
            # on_function_calls_started event handler, so the counter guard
            # prevents the timer from starting while a function call is in progress.
            if not self._user_turn_in_progress and self._function_calls_in_progress == 0:
                await self._start_idle_timer()
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._cancel_idle_timer()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_turn_in_progress = True
            await self._cancel_idle_timer()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_turn_in_progress = False
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._function_calls_in_progress += len(frame.function_calls)
            await self._cancel_idle_timer()
        elif isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame)):
            self._function_calls_in_progress = max(0, self._function_calls_in_progress - 1)

    async def _start_idle_timer(self):
        """Start (or restart) the idle timer."""
        if self._user_idle_timeout <= 0:
            return
        await self._cancel_idle_timer()
        self._idle_timer_task = self.task_manager.create_task(
            self._idle_timer_expired(),
            f"{self}::idle_timer",
        )

    async def _cancel_idle_timer(self):
        """Cancel the idle timer if running."""
        if self._idle_timer_task:
            await self.task_manager.cancel_task(self._idle_timer_task)
            self._idle_timer_task = None

    async def _idle_timer_expired(self):
        """Sleep for the timeout duration then fire the idle event."""
        await asyncio.sleep(self._user_idle_timeout)
        self._idle_timer_task = None
        await self._call_event_handler("on_user_turn_idle")
