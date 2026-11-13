#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User mute strategy that mutes the user until the bot completes its first speech."""

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
)
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

    The mute is also released if the bot's first speaking turn fails before
    producing any audio, so that a failed opening leaves the user able to speak.

    """

    def __init__(self):
        """Initialize the mute-until-first-bot-complete user mute strategy."""
        super().__init__()
        self._bot_started_speaking = False
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
            self._bot_started_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(frame)
        elif isinstance(frame, ErrorFrame):
            await self._handle_error(frame)

        return not self._first_speech_handled

    async def _handle_bot_stopped_speaking(self, frame: BotStoppedSpeakingFrame):
        if not self._first_speech_handled:
            self._first_speech_handled = True

    async def _handle_error(self, frame: ErrorFrame):
        """Release the mute when the first speaking turn fails before any audio.

        No audio means no `BotStoppedSpeakingFrame`, and no later one to wait
        for either, since a muted user can never prompt another turn. Only
        errors before the bot starts speaking count: after that the transport
        ends the turn on its own once the audio dries up.
        """
        if self._bot_started_speaking:
            return

        if not self._first_speech_handled:
            logger.warning(
                f"{self}: releasing the user mute without the bot having completed its first "
                f"speech, after an error from {frame.processor}: {frame.error}"
            )
            self._first_speech_handled = True
