#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Minimum words interruption strategy for word count-based interruptions."""

from loguru import logger

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.turns.base_interruption_strategy import BaseInterruptionStrategy


class MinWordsInterruptionStrategy(BaseInterruptionStrategy):
    """Interruption strategy based on minimum number of words spoken.

    This is an interruption strategy based on a minimum number of words said
    by the user. That is, the strategy will be true if the user has said at
    least that amount of words.
    """

    def __init__(self, *, min_words: int, use_interim: bool = True):
        """Initialize the minimum words interruption strategy.

        Args:
            min_words: Minimum number of words required to trigger an interruption.
            use_interim: Whether the strategy should use interim frames for faster interruptions.
        """
        super().__init__()
        self._min_words = min_words
        self._use_interim = use_interim
        self._text = ""

    def reset(self):
        """Reset the interruption strategy."""
        super().reset()
        self._text = ""

    async def process_frame(self, frame: Frame):
        """Process an incoming frame.

        The analysis of incoming frames will decide if the bot should be interrupted.

        Args:
            frame: The frame to be processed.
        """
        await super().process_frame(frame)

        if isinstance(frame, TranscriptionFrame) and not self._use_interim:
            await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            await self._handle_interim_transcription(frame)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Check if the transcription has enough words to interrupt.

        Args:
            frame: The transcription frame to be processed.
        """
        self._text += frame.text

        word_count = len(self._text.split())
        should_interrupt = word_count >= self._min_words

        logger.debug(
            f"{self} should_interrupt={should_interrupt} num_spoken_words={word_count} min_words={self._min_words}"
        )

        if should_interrupt:
            await self.trigger_interruption()

    async def _handle_interim_transcription(self, frame: InterimTranscriptionFrame):
        """Check if the interim transcription has enough words to interrupt.

        Args:
            frame: The interim transcription frame to be processed.
        """
        word_count = len(frame.text.split())
        should_interrupt = word_count >= self._min_words

        logger.debug(
            f"{self} interim=True should_interrupt={should_interrupt} num_spoken_words={word_count} min_words={self._min_words}"
        )

        if should_interrupt:
            await self.trigger_interruption()
