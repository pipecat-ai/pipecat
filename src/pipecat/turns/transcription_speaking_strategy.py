#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcription time-based speaking strategy."""

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.base_speaking_strategy import BaseSpeakingStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class TranscriptionSpeakingStrategy(BaseSpeakingStrategy):
    """Speaking strategy based on time after a transcription is received.

    This is a speaking strategy based on the time elapsed after a transcription
    is received.
    """

    def __init__(self, *, timeout: float = 0.4):
        """Initialize the speaking strategy.

        Args:
            timeout: Timeout to trigger the strategy after a transcription is
            received.
        """
        super().__init__()
        self._timeout = timeout
        self._text = ""
        self._vad_user_speaking = False
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def reset(self):
        """Reset the speaking strategy."""
        super().reset()
        self._text = ""
        self._vad_user_speaking = False
        self._event.clear()

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)
        self._task = task_manager.create_task(self._task_handler(), f"{self}::_task_handler")

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        if self._task:
            await self.task_manager.cancel_task(self._task)
            self._task = None

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
        self._event.set()

    async def _task_handler(self):
        """Asynchronously check if the bot should start speaking.

        If we have not received a transcription in the specified amount of time
        (and we initially received one) and the user is not speaking, then the
        bot is ready to speak.
        """
        while True:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
                self._event.clear()
            except asyncio.TimeoutError:
                if self._text and not self._vad_user_speaking:
                    await self.trigger_speech()
