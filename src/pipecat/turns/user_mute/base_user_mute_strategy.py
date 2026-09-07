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
