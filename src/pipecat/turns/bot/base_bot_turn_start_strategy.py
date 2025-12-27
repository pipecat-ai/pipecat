#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base turn start strategy for determining when the bot should start speaking."""

from typing import Optional, Type

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class BaseBotTurnStartStrategy(BaseObject):
    """Base class for strategies that determine when the bot should start speaking.

    Subclasses should implement logic to detect when the bot should start
    speaking. This could be based on analyzing incoming frames (such as
    transcriptions), conversation state, or other heuristics.

    Events triggered by bot turn start strategies:

      - `on_push_frame`: Indicates the strategy wants to push a frame.
      - `on_bot_turn_started`: Signals that the bot should start speaking.

    """

    def __init__(self, **kwargs):
        """Initialize the base bot turn start strategy."""
        super().__init__(**kwargs)
        self._task_manager: Optional[BaseTaskManager] = None
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_bot_turn_started", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} bot turn start strategy was not properly setup")
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

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to decide whether the bot should speak.

        Subclasses should override this to implement logic that decides whether
        the bot turn has started.

        Args:
            frame: The frame to be analyzed.
        """
        pass

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Emit on_push_frame to push a frame using the user aggreagtor.

        Args:
            frame: The frame to be pushed.
            direction: What direction the frame should be pushed to.
        """
        await self._call_event_handler("on_push_frame", frame, direction)

    async def broadcast_frame(self, frame_cls: Type[Frame], **kwargs):
        """Emit on_broadcast_frame to broadcast a frame using the user aggreagtor.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.
        """
        await self._call_event_handler("on_broadcast_frame", frame_cls, **kwargs)

    async def trigger_bot_turn_started(self):
        """Trigger the `on_bot_turn_started` event."""
        await self._call_event_handler("on_bot_turn_started")
