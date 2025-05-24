#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputDTMFFrame,
    KeypadEntry,
    StartFrame,
    StartInterruptionFrame,
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
    - Interruption occurs

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
        self._aggregation_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Start the aggregation task when the pipeline starts
            await self._start_aggregation_task(direction)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputDTMFFrame):
            # Push the DTMF frame first, then handle aggregation
            await self.push_frame(frame, direction)
            await self._handle_dtmf_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            # Flush on interruption
            if self._aggregation:
                await self._flush_aggregation(direction)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            # Flush any pending aggregation
            if self._aggregation:
                await self._flush_aggregation(direction)
            await self._stop_aggregation_task()
            await self.push_frame(frame, direction)
        else:
            # Push all other frames
            await self.push_frame(frame, direction)

    async def _handle_dtmf_frame(self, frame: InputDTMFFrame, direction: FrameDirection):
        # Add digit to aggregation
        digit_value = frame.button.value
        self._aggregation += digit_value

        # Check for immediate flush conditions
        if frame.button == self._termination_digit:
            await self._flush_aggregation(direction)
        else:
            # Signal new digit received
            self._digit_event.set()

    async def _start_aggregation_task(self, direction: FrameDirection):
        """Start the aggregation task."""
        if not self._aggregation_task:
            self._aggregation_task = self.create_task(self._aggregation_task_handler(direction))

    async def _aggregation_task_handler(self, direction: FrameDirection):
        """Background task that handles timeout-based flushing."""
        while True:
            try:
                await asyncio.wait_for(self._digit_event.wait(), timeout=self._idle_timeout)
                self._digit_event.clear()
            except asyncio.TimeoutError:
                if self._aggregation:
                    await self._flush_aggregation(direction)

    async def _flush_aggregation(self, direction: FrameDirection):
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

        await self.push_frame(transcription_frame, direction)

        # Reset aggregation
        self._aggregation = ""

    async def _stop_aggregation_task(self):
        """Stop the aggregation task."""
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            self._aggregation_task = None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await super().cleanup()
        await self._stop_aggregation_task()
