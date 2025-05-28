#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputDTMFFrame,
    KeypadEntry,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601


class DTMFAggregator(FrameProcessor):
    """Aggregates DTMF frames into meaningful sequences for LLM processing.

    The aggregator accumulates digits from InputDTMFFrame instances and flushes
    when:
    - Timeout occurs (configurable idle period)
    - Termination digit is received (default: '#')
    - EndFrame or CancelFrame is received

    Emits TranscriptionFrame for compatibility with existing LLM context aggregators.

    Args:
        timeout: Idle timeout in seconds before flushing
        termination_digit: Digit that triggers immediate flush
        prefix: Prefix added to DTMF sequence in transcription
    """

    def __init__(
        self,
        timeout: float = 2.0,
        termination_digit: KeypadEntry = KeypadEntry.POUND,
        prefix: str = "DTMF: ",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._aggregation = ""
        self._idle_timeout = timeout
        self._termination_digit = termination_digit
        self._prefix = prefix

        self._digit_event = asyncio.Event()
        self._aggregation_task: Optional[asyncio.Task] = None

    def _create_aggregation_task(self) -> None:
        """Creates the aggregation task if it hasn't been created yet."""
        if not self._aggregation_task:
            self._aggregation_task = self.create_task(self._aggregation_task_handler())

    async def _stop(self) -> None:
        """Stops and cleans up the aggregation task."""
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            self._aggregation_task = None

    async def cleanup(self) -> None:
        await super().cleanup()
        if self._aggregation_task:
            await self._stop()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, (EndFrame, CancelFrame)):
            # Flush any pending aggregation before stopping
            if self._aggregation:
                await self._flush_aggregation()
            await self._stop()
            await self.push_frame(frame, direction)
            return

        # Push all other frames downstream immediately
        await self.push_frame(frame, direction)

        if isinstance(frame, InputDTMFFrame):
            await self._handle_dtmf_frame(frame)

    async def _handle_dtmf_frame(
        self,
        frame: InputDTMFFrame,
    ):
        # Create task on first DTMF input if needed
        if not self._aggregation_task:
            self._create_aggregation_task()

        digit_value = frame.button.value
        self._aggregation += digit_value

        # If this is the first digit, send BotInterruptionFrame upstream
        # But use a separate task to avoid interfering with current frame processing
        if len(self._aggregation) == 1:
            # Use create_task to avoid queue issues
            self.create_task(self._send_interruption_frame())

        # Check for immediate flush conditions
        if frame.button == self._termination_digit:
            await self._flush_aggregation()
        else:
            # Signal new digit received
            self._digit_event.set()

    async def _send_interruption_frame(self):
        """Send an interruption frame in a separate task to avoid queue issues.

        This interruption frame allows for the user to interrupt the bot's speaking
        with a keypress. Without this, the TranscriptionFrame generated is accompanied
        by EmulatedUserStarted/StoppedSpeakingFrames, which the bot currently ignores.
        This treats the keypress as an explicit input which the bot should respond to.
        """
        try:
            await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)
        except Exception as e:
            print(f"Error sending interruption frame: {e}")

    async def _aggregation_task_handler(self):
        """Background task that handles timeout-based flushing."""
        try:
            while True:
                try:
                    await asyncio.wait_for(self._digit_event.wait(), timeout=self._idle_timeout)
                except asyncio.TimeoutError:
                    if self._aggregation:
                        await self._flush_aggregation()
                finally:
                    self._digit_event.clear()
        except asyncio.CancelledError:
            # Task was cancelled - exit cleanly
            pass
        except Exception as e:
            # Log unexpected errors but don't crash
            print(f"Unexpected error in DTMF aggregation task: {e}")

    async def _flush_aggregation(self):
        """Flush the current aggregation as a TranscriptionFrame."""
        if not self._aggregation:
            return

        sequence = self._aggregation

        # Create transcription with prefix for LLM context
        transcription_text = f"{self._prefix}{sequence}"

        # Create and push transcription frame
        transcription_frame = TranscriptionFrame(
            text=transcription_text, user_id="", timestamp=time_now_iso8601()
        )

        await self.push_frame(transcription_frame)

        # Reset aggregation
        self._aggregation = ""
