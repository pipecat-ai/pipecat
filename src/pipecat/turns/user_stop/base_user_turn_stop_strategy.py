#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base user turn stop strategy for determining when the user stopped speaking."""

from dataclasses import dataclass
from typing import Optional, Type

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


@dataclass
class UserTurnStoppedParams:
    """Parameters emitted when a user turn stops.

    These parameters are passed to the `on_user_turn_stopped` event and provide
    contextual information about how the end of user turn should be handled by
    the user aggregator.

    Attributes:
        enable_user_speaking_frames: Whether the user aggregator should emit
            frames indicating user speaking state (e.g., user stopped speaking).
            This is typically enabled by default, but may be disabled when another
            component (such as an STT service) is already responsible for
            generating user speaking frames.

    """

    enable_user_speaking_frames: bool


class BaseUserTurnStopStrategy(BaseObject):
    """Base class for strategies that determine when the user stops speaking.

    Subclasses should implement logic to detect when the user stops
    speaking. This could be based on analyzing incoming frames (such as
    transcriptions), conversation state, or other heuristics.

    Events triggered by strategies:

      - `on_push_frame`: Indicates the strategy wants to push a frame.
      - `on_user_turn_stopped`: Signals that the user stopped speaking.

    """

    def __init__(self, *, enable_user_speaking_frames: bool = True, **kwargs):
        """Initialize the base user turn stop strategy.

        Args:
            enable_user_speaking_frames: If True, the aggregator will emit frames
                indicating when the user stops speaking. This is enabled by default,
                but you may want to disable it if another component (e.g., an STT
                service) is already generating these frames.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._enable_user_speaking_frames = enable_user_speaking_frames
        self._task_manager: Optional[BaseTaskManager] = None
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_user_turn_stopped", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} user turn stop strategy was not properly setup")
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
        """Process an incoming frame to decide whether the user stopped speaking.

        Subclasses should override this to implement logic that decides whether
        the user has stopped speaking.

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

    async def trigger_user_turn_stopped(self):
        """Trigger the `on_user_turn_stopped` event."""
        await self._call_event_handler(
            "on_user_turn_stopped",
            UserTurnStoppedParams(enable_user_speaking_frames=self._enable_user_speaking_frames),
        )
