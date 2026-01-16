#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User mute strategy that always mutes the user while the bot is speaking."""

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


class AlwaysUserMuteStrategy(BaseUserMuteStrategy):
    """User mute strategy that always mutes the user while the bot is speaking."""

    def __init__(self):
        """Initialize the always user mute strategy."""
        super().__init__()
        self._bot_speaking = False

    async def reset(self):
        """Reset the strategy to its initial state."""
        self._bot_speaking = False

    async def process_frame(self, frame: Frame) -> bool:
        """Process an incoming frame.

        Args:
            frame: The frame to be processed.

        Returns:
            Whether the strategy is muted.
        """
        await super().process_frame(frame)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        return self._bot_speaking
