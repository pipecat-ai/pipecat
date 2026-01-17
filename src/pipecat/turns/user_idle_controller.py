#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a controller for managing user idle detection."""

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    BotSpeakingFrame,
    Frame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class UserIdleController(BaseObject):
    """Controller for managing user idle detection.

    This class monitors user activity and triggers an event when the user has been
    idle (not speaking) for a configured timeout period. It only starts monitoring
    after the first conversation activity and does not trigger while the bot is
    speaking or function calls are in progress.

    The controller tracks activity using continuous frames (UserSpeakingFrame and
    BotSpeakingFrame) which are emitted repeatedly while speaking is happening, and
    state-based tracking for function calls (FunctionCallsStartedFrame and
    FunctionCallResultFrame) which are only sent at start and end.

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
        user_idle_timeout: float,
    ):
        """Initialize the user idle controller.

        Args:
            user_idle_timeout: Timeout in seconds before considering the user idle.
        """
        super().__init__()

        self._user_idle_timeout = user_idle_timeout

        self._task_manager: Optional[BaseTaskManager] = None

        self._conversation_started = False
        self._function_call_in_progress = False

        self.user_idle_event = asyncio.Event()
        self.user_idle_task: Optional[asyncio.Task] = None

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

        if not self.user_idle_task:
            self.user_idle_task = self.task_manager.create_task(
                self.user_idle_task_handler(),
                f"{self}::user_idle_task_handler",
            )

    async def cleanup(self):
        """Cleanup the controller."""
        await super().cleanup()

        if self.user_idle_task:
            await self.task_manager.cancel_task(self.user_idle_task)
            self.user_idle_task = None

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to track user activity state.

        Args:
            frame: The frame to be processed.
        """
        # Start monitoring on first conversation activity
        if not self._conversation_started:
            if isinstance(frame, (UserStartedSpeakingFrame, BotSpeakingFrame)):
                self._conversation_started = True
                self.user_idle_event.set()
            else:
                return

        # Reset idle timer on continuous activity frames
        if isinstance(frame, (UserSpeakingFrame, BotSpeakingFrame)):
            await self._handle_activity(frame)
        # Track function call state (start/end frames, not continuous)
        elif isinstance(frame, FunctionCallsStartedFrame):
            await self._handle_function_calls_started(frame)
        elif isinstance(frame, FunctionCallResultFrame):
            await self._handle_function_call_result(frame)

    async def _handle_activity(self, _: UserSpeakingFrame | BotSpeakingFrame):
        """Handle continuous activity frames that should reset the idle timer.

        These frames are emitted continuously while the user or bot is speaking,
        so we simply reset the timer whenever we receive them.

        Args:
            frame: The activity frame to process.
        """
        self.user_idle_event.set()

    async def _handle_function_calls_started(self, _: FunctionCallsStartedFrame):
        """Handle function calls started event.

        Function calls can take longer than the timeout, so we track their state
        to prevent idle callbacks while they're in progress.

        Args:
            frame: The FunctionCallsStartedFrame to process.
        """
        self._function_call_in_progress = True
        self.user_idle_event.set()

    async def _handle_function_call_result(self, _: FunctionCallResultFrame):
        """Handle function call result event.

        Args:
            frame: The FunctionCallResultFrame to process.
        """
        self._function_call_in_progress = False
        self.user_idle_event.set()

    async def user_idle_task_handler(self):
        """Monitors for idle timeout and triggers events.

        Runs in a loop until cancelled. The idle timer is reset whenever activity
        frames are received (UserSpeakingFrame or BotSpeakingFrame). Function calls
        are tracked via state since they only send start/end frames. If no activity
        is detected for the configured timeout period and no function call is in
        progress, the on_user_turn_idle event is triggered.
        """
        while True:
            try:
                await asyncio.wait_for(self.user_idle_event.wait(), timeout=self._user_idle_timeout)
                self.user_idle_event.clear()
            except asyncio.TimeoutError:
                # Only trigger if conversation has started and no function call is in progress
                if self._conversation_started and not self._function_call_in_progress:
                    await self._call_event_handler("on_user_turn_idle")
