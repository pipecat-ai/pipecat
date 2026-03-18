#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base filter for deciding whether user frames should be filtered."""

from typing import Optional

from pipecat.frames.frames import Frame
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class BaseUserFrameFilter(BaseObject):
    """Base class for filters that decide whether user frames should be blocked.

    A user frame filter examines incoming frames and decides whether each frame
    should pass through to the aggregator or be blocked. Unlike mute strategies
    which suppress all user frames based on system state, filters can selectively
    block frames based on content (e.g., waiting for a wake phrase).

    The filter is evaluated per frame and returns a boolean indicating whether
    the frame should pass through (True) or be blocked (False).

    """

    def __init__(self, **kwargs):
        """Initialize the base user frame filter."""
        super().__init__(**kwargs)
        self._task_manager: Optional[BaseTaskManager] = None

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} user frame filter was not properly setup")
        return self._task_manager

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the filter with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        self._task_manager = task_manager

    async def cleanup(self):
        """Cleanup the filter."""
        pass

    async def reset(self):
        """Reset the filter to its initial state."""
        pass

    async def process_frame(self, frame: Frame) -> bool:
        """Process an incoming frame.

        Args:
            frame: The frame to be processed.

        Returns:
            True if the frame should pass through, False if it should be blocked.
        """
        return True
