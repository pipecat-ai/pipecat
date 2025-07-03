#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gated OpenAI LLM context aggregator for controlled message flow."""

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.sync.base_notifier import BaseNotifier


class GatedOpenAILLMContextAggregator(FrameProcessor):
    """Aggregator that gates OpenAI LLM context frames until notified.

    This aggregator captures OpenAI LLM context frames and holds them until
    a notifier signals that they can be released. This is useful for controlling
    the flow of context frames based on external conditions or timing.
    """

    def __init__(self, *, notifier: BaseNotifier, start_open: bool = False, **kwargs):
        """Initialize the gated context aggregator.

        Args:
            notifier: The notifier that controls when frames are released.
            start_open: If True, the first context frame passes through immediately.
            **kwargs: Additional arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._notifier = notifier
        self._start_open = start_open
        self._last_context_frame = None
        self._gate_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, gating OpenAI LLM context frames.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame)
            await self._start()
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame)
        elif isinstance(frame, OpenAILLMContextFrame):
            if self._start_open:
                self._start_open = False
                await self.push_frame(frame, direction)
            else:
                self._last_context_frame = frame
        else:
            await self.push_frame(frame, direction)

    async def _start(self):
        """Start the gate task handler."""
        if not self._gate_task:
            self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        """Stop the gate task handler."""
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

    async def _gate_task_handler(self):
        """Handle the gating logic by waiting for notifications and releasing frames."""
        while True:
            await self._notifier.wait()
            if self._last_context_frame:
                await self.push_frame(self._last_context_frame)
                self._last_context_frame = None
