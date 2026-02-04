#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcription time-based user turn stop strategy."""

import asyncio
import time
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    SpeechControlParamsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class TranscriptionUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy based on transcriptions.

    This strategy triggers the user turn stopped event based on two factors:

    1. STT timeout: The P99 time for the STT service to return a transcription
       after the user stops speaking. This is adjusted by the VAD stop_secs.

    2. User resume speaking timeout: Time to wait for the user to potentially
       say more after they pause.

    The timeout starts when VADUserStoppedSpeakingFrame is received. For services
    that support finalization (TranscriptionFrame.finalized=True), the turn can
    be triggered immediately once the finalized transcript is received and the
    user resume speaking timeout has elapsed.

    """

    def __init__(
        self, *, user_resume_speaking_timeout: float = 0.6, _stt_timeout: float = 0.5, **kwargs
    ):
        """Initialize the transcription-based user turn stop strategy.

        Args:
            user_resume_speaking_timeout: Time to wait for the user to potentially
                say more after they pause speaking. Defaults to 0.5 seconds.
            _stt_timeout: Internal P99 placeholder for STT latency. This is not
                intended to be set by users - it will be wired to service-specific
                values later. Defaults to 0.5 seconds.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._user_resume_speaking_timeout = user_resume_speaking_timeout
        self._stt_timeout: float = _stt_timeout
        self._stop_secs: float = 0.0  # VAD stop_secs from SpeechControlParamsFrame

        self._text = ""
        self._vad_user_speaking = False
        self._transcript_finalized = False
        self._vad_stopped_time: Optional[float] = None
        self._timeout_task: Optional[asyncio.Task] = None

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""
        self._vad_user_speaking = False
        self._transcript_finalized = False
        self._vad_stopped_time = None

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to update strategy state.

        Updates internal transcription text and VAD state. The user end turn
        will be triggered when appropriate based on the collected frames.

        Args:
            frame: The frame to be analyzed.

        """
        if isinstance(frame, SpeechControlParamsFrame):
            await self._handle_speech_control_params(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

    async def _handle_speech_control_params(self, frame: SpeechControlParamsFrame):
        """Handle speech control parameters to extract VAD stop_secs.

        Args:
            frame: The speech control parameters frame.
        """
        if frame.vad_params:
            self._stop_secs = frame.vad_params.stop_secs

    async def _handle_vad_user_started_speaking(self, _: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._vad_user_speaking = True
        self._transcript_finalized = False
        self._vad_stopped_time = None
        # Cancel any pending timeout
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

    async def _handle_vad_user_stopped_speaking(self, _: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False
        self._vad_stopped_time = time.time()

        # Start the timeout task
        timeout = self._calculate_timeout()
        self._timeout_task = self.task_manager.create_task(
            self._timeout_handler(timeout), f"{self}::_timeout_handler"
        )

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        self._text += frame.text
        if frame.finalized:
            self._transcript_finalized = True
            # For finalized transcripts, check if we can trigger early
            await self._maybe_trigger_user_turn_stopped()

        # Fallback: handle transcripts when no VAD stop was received.
        # This handles edge cases where transcripts arrive without VAD firing.
        # _vad_stopped_time is None means VAD stopped hasn't been received yet.
        # In fallback mode, reset timeout on each transcript to wait for inactivity.
        if not self._vad_user_speaking and self._vad_stopped_time is None:
            # Cancel existing fallback timeout if any
            if self._timeout_task:
                await self.task_manager.cancel_task(self._timeout_task)
            timeout = self._calculate_timeout()
            self._timeout_task = self.task_manager.create_task(
                self._timeout_handler(timeout), f"{self}::_timeout_handler"
            )

    def _calculate_timeout(self) -> float:
        """Calculate the timeout value based on current state.

        Returns:
            The timeout in seconds to wait after VAD stopped speaking.
        """
        # Adjust STT timeout by VAD stop_secs since that time has already elapsed
        effective_stt_wait = max(0, self._stt_timeout - self._stop_secs)

        # If transcript is already finalized, we don't need to wait for STT
        if self._transcript_finalized:
            return self._user_resume_speaking_timeout

        return max(effective_stt_wait, self._user_resume_speaking_timeout)

    async def _timeout_handler(self, timeout: float):
        """Wait for the timeout then trigger user turn stopped if conditions met.

        Args:
            timeout: The timeout in seconds to wait.
        """
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        finally:
            self._timeout_task = None

        await self._maybe_trigger_user_turn_stopped()

    async def _maybe_trigger_user_turn_stopped(self):
        """Trigger user turn stopped if conditions are met.

        Conditions:
        - User is not currently speaking
        - We have transcription text
        - Either the timeout has elapsed OR we have a finalized transcript
          and user_resume_speaking_timeout has elapsed
        """
        if self._vad_user_speaking or not self._text:
            return

        # For finalized transcripts, check if user_resume_speaking_timeout has elapsed.
        # If elapsed, trigger user turn stopped immediately. Else, wait for user resume
        # speaking timeout.
        if self._transcript_finalized and self._vad_stopped_time is not None:
            elapsed = time.time() - self._vad_stopped_time
            if elapsed >= self._user_resume_speaking_timeout:
                # Cancel any remaining timeout since we're triggering now
                if self._timeout_task:
                    await self.task_manager.cancel_task(self._timeout_task)
                    self._timeout_task = None
                await self.trigger_user_turn_stopped()
                return

        # For non-finalized, only trigger if timeout task has completed
        if self._timeout_task is None:
            await self.trigger_user_turn_stopped()
