#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base strategy for deciding whether user frames should be muted."""

from typing import Optional

from pipecat.frames.frames import Frame
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class BaseUserMuteStrategy(BaseObject):
    """Base class for strategies that decide whether user frames should be muted.

    A user mute strategy determines whether incoming user frames should be
    suppressed based on the *current system state*.

    Typical heuristics include:
    - The bot is currently speaking, so user should be muted
    - A function call or tool execution is in progress
    - The system is otherwise not ready to accept user input

    The strategy is evaluated per frame and returns a boolean indicating whether
    the user should be muted.

    """

    def __init__(self, **kwargs):
        """Initialize the base user mute strategy."""
        super().__init__(**kwargs)
        self._task_manager: Optional[BaseTaskManager] = None

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} user mute strategy was not properly setup")
        return self._task_manager

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        self._task_manager = task_manager

    async def cleanup(self):
        """Cleanup the strategy."""
        pass

    async def reset(self):
        """Reset the strategy to its initial state."""
        pass

    async def process_frame(self, frame: Frame) -> bool:
        """Process an incoming frame.

        Args:
            frame: The frame to be processed.

        Returns:
            Whether the strategy is muted.
        """
        return False
