#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base speaking strategy for determining when the bot can start speaking."""

from typing import Optional

from pipecat.frames.frames import Frame
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class BaseSpeakingStrategy(BaseObject):
    """Base class for speaking strategies.

    This is a base class for speaking strategies. Speaking strategies decide
    when the bot is ready to start talking. For example, there could be
    strategies based on time or strategies based on turn-detection models.

    Speaking strategies can trigger the following events:

      - on_push_frame
      - on_should_speak

    The `on_push_frame` event is used to indicate that the strategy would like
    to push a frame.

    The `on_should_speak` event is used to asynchronously indicate the bot
    should start speaking.
    """

    def __init__(self, **kwargs):
        """Initialize the base speaking strategy."""
        super().__init__(**kwargs)
        self._task_manager: Optional[BaseTaskManager] = None
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_should_speak", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configure task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} speaking strategy was not properly setup")
        return self._task_manager

    def reset(self):
        """Reset the speaking strategy."""
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

        The analysis of incoming frames will decide if the bot should start
        speaking.

        Args:
            frame: The frame to be processed.

        """
        pass

    async def trigger_speech(self):
        await self._call_event_handler("on_should_speak")
