#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Preemptive user turn stop strategy for low-latency response generation."""

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class PreemptiveUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy that triggers LLM generation as soon as possible.

    This strategy triggers the end of the user's turn as soon as VAD detects
    silence and any transcription text is available, without waiting for ML
    turn analysis or finalized transcripts. The existing interruption mechanism
    handles cancellation if the user resumes speaking.

    Trigger conditions (any one sufficient):

    - **VAD-first:** VAD has stopped and any transcription text exists — trigger
      immediately.
    - **STT-first:** Transcription arrives while user is not speaking (no VAD) —
      trigger after a fallback STT timeout.
    - **Fallback:** VAD stopped but no transcript arrives — wait for the STT P99
      timeout, then trigger.
    """

    def __init__(self, **kwargs):
        """Initialize the preemptive user turn stop strategy.

        Args:
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._stt_timeout: float = 0.0
        self._stop_secs: float = 0.0
        self._text: str = ""
        self._vad_user_speaking: bool = False
        self._vad_stopped: bool = False
        self._has_triggered: bool = False
        self._timeout_task: Optional[asyncio.Task] = None

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""
        self._vad_user_speaking = False
        self._vad_stopped = False
        self._has_triggered = False

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

        Args:
            frame: The frame to be analyzed.
        """
        if isinstance(frame, STTMetadataFrame):
            self._stt_timeout = frame.ttfs_p99_latency
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

    async def _handle_vad_user_started_speaking(self, _: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._vad_user_speaking = True
        self._vad_stopped = False
        # Cancel any pending timeout
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False
        self._vad_stopped = True
        self._stop_secs = frame.stop_secs

        # Try to trigger immediately if we already have text
        await self._maybe_trigger_user_turn_stopped()

        # If we haven't triggered yet (no text), start a fallback timeout
        if not self._has_triggered:
            timeout = max(0, self._stt_timeout - self._stop_secs)
            self._timeout_task = self.task_manager.create_task(
                self._timeout_handler(timeout), f"{self}::_timeout_handler"
            )

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        self._text += frame.text

        if self._has_triggered:
            return

        # VAD path: if VAD has stopped and we have text, trigger immediately
        if self._vad_stopped:
            await self._trigger()
            return

        # No-VAD fallback: start/reset timeout on each transcription.
        # The timeout handler will trigger when it completes.
        if not self._vad_user_speaking:
            if self._timeout_task:
                await self.task_manager.cancel_task(self._timeout_task)
            timeout = max(0, self._stt_timeout - self._stop_secs)
            self._timeout_task = self.task_manager.create_task(
                self._timeout_handler(timeout), f"{self}::_timeout_handler"
            )

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

        if not self._has_triggered and not self._vad_user_speaking and self._text:
            await self._trigger()

    async def _trigger(self):
        """Trigger user turn stopped and cancel any pending timeout."""
        self._has_triggered = True
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None
        await self.trigger_user_turn_stopped()

    async def _maybe_trigger_user_turn_stopped(self):
        """Trigger user turn stopped if VAD has stopped and text exists."""
        if self._has_triggered or self._vad_user_speaking or not self._text:
            return

        if self._vad_stopped:
            await self._trigger()
