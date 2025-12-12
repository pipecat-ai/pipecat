#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcription time-based speaking strategy."""

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.base_speaking_strategy import BaseSpeakingStrategy


class EagerTranscriptionSpeakingStrategy(BaseSpeakingStrategy):
    """Speaking strategy based on final transcriptions.

    This speaking strategy uses final transcription frames to indicate the bot
    should start speaking. This is the faster speaking strategy, however it
    might cause issues if STT service return two consecutive final
    transcriptions.

    """

    def __init__(self):
        """Initialize the speaking strategy."""
        super().__init__()
        self._text = ""
        self._vad_user_speaking = False

    async def reset(self):
        """Reset the speaking strategy."""
        await super().reset()
        self._text = ""
        self._vad_user_speaking = False

    async def process_frame(self, frame: Frame):
        """Process an incoming frame.

        The analysis of incoming frames will decide if the bot should start
        speaking.

        Args:
            frame: The frame to be processed.

        """
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._vad_user_speaking = True

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        self._text += frame.text
        if not self._vad_user_speaking:
            await self.trigger_speech()
