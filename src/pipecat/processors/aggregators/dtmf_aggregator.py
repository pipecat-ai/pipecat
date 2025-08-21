#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DTMF aggregation processor for converting keypad input to transcription.

This module provides a frame processor that aggregates DTMF (Dual-Tone Multi-Frequency)
keypad inputs into meaningful sequences and converts them to transcription frames
for downstream processing by LLM context aggregators.
"""

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputDTMFFrame,
    KeypadEntry,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.time import time_now_iso8601


class DTMFAggregator(FrameProcessor):
    """Aggregates DTMF frames into meaningful sequences for LLM processing.

    The aggregator accumulates digits from InputDTMFFrame instances and flushes
    when:

    - Timeout occurs (configurable idle period)
    - Termination digit is received (default: '#')
    - EndFrame or CancelFrame is received

    Emits TranscriptionFrame for compatibility with existing LLM context aggregators.
    """

    def __init__(
        self,
        timeout: float = 2.0,
        termination_digit: KeypadEntry = KeypadEntry.POUND,
        prefix: str = "DTMF: ",
        **kwargs,
    ):
        """Initialize the DTMF aggregator.

        Args:
            timeout: Idle timeout in seconds before flushing
            termination_digit: Digit that triggers immediate flush
            prefix: Prefix added to DTMF sequence in transcription
            **kwargs: Additional arguments passed to FrameProcessor
        """
        super().__init__(**kwargs)
        self._aggregation = ""
        self._idle_timeout = timeout
        self._termination_digit = termination_digit
        self._prefix = prefix

        self._digit_event = asyncio.Event()
        self._aggregation_task: Optional[asyncio.Task] = None

    async def cleanup(self) -> None:
        """Clean up resources."""
        await super().cleanup()
        await self._stop_aggregation_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames and handle DTMF aggregation.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._create_aggregation_task()
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            if self._aggregation:
                await self._flush_aggregation()
            await self._stop_aggregation_task()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputDTMFFrame):
            # Push the DTMF frame downstream first
            await self.push_frame(frame, direction)
            # Then handle it in order for the TranscriptionFrame to be emitted
            # after the InputDTMFFrame
            await self._handle_dtmf_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def _handle_dtmf_frame(self, frame: InputDTMFFrame):
        """Handle DTMF input frame."""
        is_first_digit = not self._aggregation

        digit_value = frame.button.value
        self._aggregation += digit_value

        # For first digit, schedule interruption in separate task
        if is_first_digit:
            await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

        # Check for immediate flush conditions
        if frame.button == self._termination_digit:
            await self._flush_aggregation()
        else:
            # Signal digit received for timeout handling
            self._digit_event.set()

    def _create_aggregation_task(self) -> None:
        """Creates the aggregation task if it hasn't been created yet."""
        if not self._aggregation_task:
            self._aggregation_task = self.create_task(self._aggregation_task_handler())

    async def _stop_aggregation_task(self) -> None:
        """Stops the aggregation task."""
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            self._aggregation_task = None

    async def _aggregation_task_handler(self):
        """Background task that handles timeout-based flushing."""
        while True:
            try:
                await asyncio.wait_for(self._digit_event.wait(), timeout=self._idle_timeout)
                self._digit_event.clear()
            except asyncio.TimeoutError:
                if self._aggregation:
                    await self._flush_aggregation()

    async def _flush_aggregation(self):
        """Flush the current aggregation as a TranscriptionFrame."""
        if not self._aggregation:
            return

        sequence = self._aggregation
        transcription_text = f"{self._prefix}{sequence}"

        transcription_frame = TranscriptionFrame(
            text=transcription_text, user_id="", timestamp=time_now_iso8601()
        )
        await self.push_frame(transcription_frame)

        self._aggregation = ""
