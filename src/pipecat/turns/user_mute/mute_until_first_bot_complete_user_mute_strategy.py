#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User mute strategy that mutes the user until the bot completes its first speech."""

from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


class MuteUntilFirstBotCompleteUserMuteStrategy(BaseUserMuteStrategy):
    """User mute strategy that mutes the user until the bot completes its first speech.

    This strategy mutes user frames immediately from the start of the
    interaction, even if the bot has not started speaking yet. User input
    remains muted until the bot finishes its first speaking turn.

    After the bot completes its initial speech, all subsequent user frames are
    allowed to pass through without muting.

    Use this strategy when the bot must fully control the beginning of the
    interaction and deliver its first response without any user interruption.

    """

    def __init__(self):
        """Initialize the mute-until-first-bot-complete user mute strategy."""
        super().__init__()
        self._first_speech_handled = False

    async def reset(self):
        """Reset the strategy to its initial state."""
        self._first_speech_handled = False

    async def process_frame(self, frame: Frame) -> bool:
        """Process an incoming frame.

        Args:
            frame: The frame to be processed.

        Returns:
            Whether the strategy is muted.
        """
        await super().process_frame(frame)

        if isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(frame)

        return not self._first_speech_handled

    async def _handle_bot_stopped_speaking(self, frame: BotStoppedSpeakingFrame):
        if not self._first_speech_handled:
            self._first_speech_handled = True
