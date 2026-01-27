#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcription time-based user turn stop strategy."""

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class TranscriptionUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy based on transcriptions.

    This strategy assumes the user stops speaking once a transcription has been
    received. It handles multiple or delayed transcription frames gracefully.

    """

    def __init__(self, *, timeout: float = 0.5, **kwargs):
        """Initialize the transcription-based user turn stop strategy.

        Args:
            timeout: A short delay used internally to handle consecutive or
                slightly delayed transcriptions.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._timeout = timeout
        self._text = ""
        self._vad_user_speaking = False
        self._seen_interim_results = False
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""
        self._vad_user_speaking = False
        self._seen_interim_results = False
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
        """Process an incoming frame to update strategy state.

        Updates internal transcription text and VAD state. The user end turn
        will be triggered when appropriate based on the collected frames.

        Args:
            frame: The frame to be analyzed.

        """
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._handle_interim_transcription(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

    async def _handle_vad_user_started_speaking(self, _: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._vad_user_speaking = True

    async def _handle_vad_user_stopped_speaking(self, _: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False
        await self._maybe_trigger_user_turn_stopped()

    async def _handle_interim_transcription(self, frame: InterimTranscriptionFrame):
        self._seen_interim_results = True

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        self._text += frame.text
        # We just got a final result, so let's reset interim results.
        self._seen_interim_results = False
        # Reset aggregation timer.
        self._event.set()

    async def _task_handler(self):
        """Asynchronously monitor transcriptions and trigger user end turn when ready.

        If transcription text exists and the user is not currently speaking,
        triggers the user end turn. Handles multiple or delayed transcriptions
        gracefully.

        """
        while True:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
                self._event.clear()
            except asyncio.TimeoutError:
                await self._maybe_trigger_user_turn_stopped()

    async def _maybe_trigger_user_turn_stopped(self):
        if not self._vad_user_speaking and not self._seen_interim_results and self._text:
            await self.trigger_user_turn_stopped()
