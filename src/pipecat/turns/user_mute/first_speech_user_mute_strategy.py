#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User mute strategy that mutes the user only during the bot’s first speech."""

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


class FirstSpeechUserMuteStrategy(BaseUserMuteStrategy):
    """User mute strategy that mutes the user only during the bot’s first speech.

    This strategy allows user input before the bot starts speaking. Once the bot
    begins its first speaking turn, user frames are muted until the bot finishes
    that speech. After the bot completes its first speaking turn, user input is
    no longer muted by this strategy.

    Use this strategy when early user input is acceptable, but interruptions
    during the bot’s initial response should be prevented.

    """

    def __init__(self):
        """Initialize the first-bot-speech user mute strategy."""
        super().__init__()
        self._bot_speaking = False
        self._first_speech_handled = False

    async def reset(self):
        """Reset the strategy to its initial state."""
        self._bot_speaking = False
        self._first_speech_handled = False

    async def process_frame(self, frame: Frame) -> bool:
        """Process an incoming frame.

        Args:
            frame: The frame to be processed.

        Returns:
            Whether the strategy is muted.
        """
        await super().process_frame(frame)

        if isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(frame)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(frame)

        if self._bot_speaking and not self._first_speech_handled:
            return True

        return False

    async def _handle_bot_started_speaking(self, frame: BotStartedSpeakingFrame):
        self._bot_speaking = True

    async def _handle_bot_stopped_speaking(self, frame: BotStoppedSpeakingFrame):
        self._bot_speaking = False
        if not self._first_speech_handled:
            self._first_speech_handled = True
