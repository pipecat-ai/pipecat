#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy based on transcriptions."""

from pipecat.frames.frames import BotStartedSpeakingFrame, Frame, TranscriptionFrame
from pipecat.turns.user.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class TranscriptionUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy based on transcriptions.

    This strategy signals the start of a user turn when a transcription is
    received while the bot is speaking. It is useful as a fallback in scenarios
    where VAD-based detection fails (for example, when the user speaks very
    softly) but the STT service still produces transcriptions.

    """

    def __init__(self):
        """Initialize the base interruption strategy."""
        super().__init__()
        self._bot_speaking = False

    async def reset(self):
        """Reset the interruption strategy."""
        await super().reset()
        self._bot_speaking = False

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to detect the start of a user turn.

        Args:
            frame: The frame to be processed.
        """
        await super().process_frame(frame)

        if isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

    async def _handle_bot_started_speaking(self, _: BotStartedSpeakingFrame):
        self._bot_speaking = True

    async def _handle_transcription(self, _: TranscriptionFrame):
        if self._bot_speaking:
            await self.trigger_user_turn_started()
