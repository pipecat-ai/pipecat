#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base turn start strategy for determining when the user starts speaking."""

from typing import Optional

from pipecat.frames.frames import Frame
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class BaseUserTurnStartStrategy(BaseObject):
    """Base class for strategies that determine when a user starts speaking.

    Subclasses should implement logic to detect the start of a user's turn.
    This could be based on voice activity, number of words spoken, or other
    heuristics.

    Events triggered by user turn start strategies:

      - `on_push_frame`: Indicates the strategy wants to push a frame.
      - `on_user_turn_started`: Signals that a user turn has started.
    """

    def __init__(self, **kwargs):
        """Initialize the base user turn start strategy."""
        super().__init__(**kwargs)
        self._task_manager: Optional[BaseTaskManager] = None
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_user_turn_started", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} user turn start strategy was not properly setup")
        return self._task_manager

    async def reset(self):
        """Reset the strategy to its initial state."""
        pass

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        self._task_manager = task_manager

    async def cleanup(self):
        """Cleanup the strategy."""
        pass

    async def process_frame(self, frame: Frame):
        """Process an incoming frame.

        Subclasses should override this to implement logic that decides whether
        the user turn has started.

        Args:
            frame: The frame to be processed.

        """
        pass

    async def trigger_user_turn_started(self):
        """Trigger the `on_user_turn_started` event."""
        await self._call_event_handler("on_user_turn_started")
