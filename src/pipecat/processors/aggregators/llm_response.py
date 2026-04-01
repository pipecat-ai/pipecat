#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM response aggregator for collecting complete LLM responses."""

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class LLMFullResponseAggregator(FrameProcessor):
    """Aggregates complete LLM responses between start and end frames.

    This aggregator collects LLM text frames (tokens) received between
    `LLMFullResponseStartFrame` and `LLMFullResponseEndFrame` and provides
    the complete response via an event handler.

    The aggregator provides an "on_completion" event that fires when a full
    completion is available::

        @aggregator.event_handler("on_completion")
        async def on_completion(
            aggregator: LLMFullResponseAggregator,
            completion: str,
            completed: bool,
        ):
            # Handle the completion
            pass
    """

    def __init__(self, **kwargs):
        """Initialize the LLM full response aggregator.

        Args:
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)

        self._aggregation = ""
        self._started = False

        self._register_event_handler("on_completion")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and aggregate LLM text content.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._call_event_handler("on_completion", self._aggregation, False)
            self._aggregation = ""
            self._started = False
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)

        await self.push_frame(frame, direction)

    async def _handle_llm_start(self, _: LLMFullResponseStartFrame):
        self._started = True

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame):
        await self._call_event_handler("on_completion", self._aggregation, True)
        self._started = False
        self._aggregation = ""

    async def _handle_llm_text(self, frame: TextFrame):
        if not self._started:
            return
        self._aggregation += frame.text
