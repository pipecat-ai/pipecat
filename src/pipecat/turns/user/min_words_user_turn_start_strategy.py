#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy based on a minimum number of words spoken by the user."""

from loguru import logger

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.turns.user.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class MinWordsUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy based on a minimum number of words spoken by the user.

    This strategy signals the start of a user turn once the user has spoken at
    least a specified number of words, as determined from transcription frames.
    Optionally, interim transcriptions can be used for earlier detection.

    """

    def __init__(self, *, min_words: int, use_interim: bool = True):
        """Initialize the minimum words bot turn start strategy.

        Args:
            min_words: Minimum number of spoken words required to trigger the
                start of a user turn.
            use_interim: Whether to consider interim transcription frames for
                earlier detection.
        """
        super().__init__()
        self._min_words = min_words
        self._use_interim = use_interim
        self._text = ""

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to detect the start of a user turn.

        This method updates internal state based on transcription frames and
        triggers the user turn once the minimum word count is reached.

        Args:
            frame: The frame to be analyzed.
        """
        await super().process_frame(frame)

        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            await self._handle_interim_transcription(frame)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle a completed transcription frame and check word count.

        Args:
            frame: The transcription frame to be processed.
        """
        self._text += frame.text

        word_count = len(self._text.split())
        should_trigger = word_count >= self._min_words

        logger.debug(
            f"{self} should_trigger={should_trigger} num_spoken_words={word_count} min_words={self._min_words}"
        )

        if should_trigger:
            await self.trigger_user_turn_started()

    async def _handle_interim_transcription(self, frame: InterimTranscriptionFrame):
        """Handle an interim transcription frame and check word count.

        Args:
            frame: The interim transcription frame to be processed.
        """
        word_count = len(frame.text.split())
        should_trigger = word_count >= self._min_words

        logger.debug(
            f"{self} interim=True should_trigger={should_trigger} num_spoken_words={word_count} min_words={self._min_words}"
        )

        if should_trigger:
            await self.trigger_user_turn_started()
