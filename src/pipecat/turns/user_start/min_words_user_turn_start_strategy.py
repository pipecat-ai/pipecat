#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy based on a minimum number of words spoken by the user."""

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class MinWordsUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy based on a minimum number of words spoken by the user.

    This strategy signals the start of a user turn once the user has spoken at
    least a specified number of words, as determined from transcription frames.
    Optionally, interim transcriptions can be used for earlier detection.

    """

    def __init__(self, *, min_words: int, use_interim: bool = True, **kwargs):
        """Initialize the minimum words bot turn start strategy.

        Args:
            min_words: Minimum number of spoken words required to trigger the
                start of a user turn.
            use_interim: Whether to consider interim transcription frames for
                earlier detection.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._min_words = min_words
        self._use_interim = use_interim
        self._bot_speaking = False

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._bot_speaking = False

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to detect the start of a user turn.

        This method updates internal state based on transcription frames and
        triggers the user turn once the minimum word count is reached.

        Args:
            frame: The frame to be analyzed.
        """
        await super().process_frame(frame)

        if isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(frame)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            await self._handle_transcription(frame)

    async def _handle_bot_started_speaking(self, frame: BotStartedSpeakingFrame):
        """Handle bot started speaking frame.

        If the bot is speaking we want to interrupt using min words.

        Args:
            frame: The frame to be processed.
        """
        self._bot_speaking = True

    async def _handle_bot_stopped_speaking(self, frame: BotStoppedSpeakingFrame):
        """Handle bot started speaking frame.

        If the bot is not speaking we want to interrupt if we get a single word.

        Args:
            frame: The frame to be processed.
        """
        self._bot_speaking = False

    async def _handle_transcription(self, frame: TranscriptionFrame | InterimTranscriptionFrame):
        """Handle a completed transcription frame and check word count.

        Args:
            frame: The transcription frame to be processed.
        """
        min_words = self._min_words if self._bot_speaking else 1

        word_count = len(frame.text.split())
        should_trigger = word_count >= min_words
        is_interim = isinstance(frame, InterimTranscriptionFrame)

        logger.debug(
            f"{self} should_trigger={should_trigger} num_spoken_words={word_count} "
            f"min_words={min_words} bot_speaking={self._bot_speaking} interim_transcription={is_interim}"
        )

        if should_trigger:
            await self.trigger_user_turn_started()
