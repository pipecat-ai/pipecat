#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base strategy for deciding whether user frames should be muted."""

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessorSetup
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

    A strategy that mutes the user at a turn boundary, such as from the moment
    the user's turn ends, overrides :meth:`handle_user_turn_started` or
    :meth:`handle_user_turn_stopped`. The user aggregator calls them when the
    turn starts or stops and applies the result before the next frame, which a
    decision made in :meth:`process_frame` can't do: it takes effect from the
    frame after the one that triggered it.
    """

    def __init__(self, **kwargs):
        """Initialize the base user mute strategy."""
        super().__init__(**kwargs)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the strategy.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup.task_manager)

    async def cleanup(self):
        """Cleanup the strategy."""
        pass

    async def process_frame(self, frame: Frame) -> bool:
        """Process an incoming frame.

        Args:
            frame: The frame to be processed.

        Returns:
            Whether the strategy is muted.
        """
        return False

    async def handle_user_turn_started(self) -> bool | None:
        """Handle the start of a user turn.

        Returns:
            Whether the strategy is muted now, or None to keep the decision it
            made on the last frame.
        """
        return None

    async def handle_user_turn_stopped(self) -> bool | None:
        """Handle the end of a user turn.

        Returns:
            Whether the strategy is muted now, or None to keep the decision it
            made on the last frame.
        """
        return None
