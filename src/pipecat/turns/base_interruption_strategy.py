#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base interruption strategy for determining when users can interrupt bot speech."""

from typing import Optional

from pipecat.frames.frames import Frame
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class BaseInterruptionStrategy(BaseObject):
    """Base class for interruption strategies.

    This is a base class for interruption strategies. Interruption strategies
    decide when the user can interrupt the bot while the bot is speaking. For
    example, there could be strategies based on the number of words the user
    spoke or strategies that check how long the user has been speaking.

    Interruption strategies can trigger the following events:

      - on_push_frame
      - on_should_interrupt

    The `on_push_frame` event is used to indicate that the strategy would like
    to push a frame.

    The `on_should_interrupt` event is used to asynchronously indicate the bot
    should be interrupted.
    """

    def __init__(self, **kwargs):
        """Initialize the base interruption strategy."""
        super().__init__(**kwargs)
        self._task_manager: Optional[BaseTaskManager] = None
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_should_interrupt", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configure task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} interruption strategy was not properly setup")
        return self._task_manager

    def reset(self):
        """Reset the interruption strategy."""
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

        The analysis of incoming frames will decide if the bot should be interrupted.

        Args:
            frame: The frame to be processed.
        """
        pass

    async def trigger_interruption(self):
        await self._call_event_handler("on_should_interrupt")
