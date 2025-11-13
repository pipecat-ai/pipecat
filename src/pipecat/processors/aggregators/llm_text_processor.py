#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM text processor module for processing and aggregating raw LLM output text.

This processor will convert LLMTextFrames into AggregatedTextFrames based on the
configured text aggregator. Using the customizable aggregator, it provides
functionality to handle or manipulate LLM text frames before they are sent to other
components such as TTS services or context aggregators. It can be used to pre-aggregate
and categorize, modify, or filter direct output tokens from the LLM.
"""

from typing import Optional

from pipecat.frames.frames import (
    AggregatedTextFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class LLMTextProcessor(FrameProcessor):
    """A processor for handling or manipulating LLM text frames before they are processed further.

    This processor will convert LLMTextFrames into AggregatedTextFrames based on the configured
    text aggregator. Using the customizable aggregator, it provides functionality to handle or
    manipulate LLM text frames before they are sent to other components such as TTS services or
    context aggregators. It can be used to pre-aggregate and categorize, modify, or filter direct
    output tokens from the LLM.
    """

    def __init__(self, *, text_aggregator: Optional[BaseTextAggregator] = None, **kwargs):
        """Initialize the LLM text processor.

        Args:
            text_aggregator: An optional text aggregator to use for processing LLM text frames. By
                default, a SimpleTextAggregator aggregating by sentence will be used.
            **kwargs: Additional arguments passed to parent class.

        TODO: Allow transformations per aggregation type or all (and deprecate the TTS filters).
        """
        super().__init__(**kwargs)
        self._text_aggregator: BaseTextAggregator = text_aggregator or SimpleTextAggregator()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an LLMTextFrames using the aggregator to generate AggregatedTextFrames.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame.skip_tts)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._handle_llm_end()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_interruption(self, _):
        """Handle interruptions by resetting the text aggregator."""
        await self._text_aggregator.handle_interruption()

    async def reset(self):
        """Reset the internal state of the text processor and its aggregator."""
        await self._text_aggregator.reset()

    async def _handle_llm_text(self, in_frame: LLMTextFrame):
        aggregation = await self._text_aggregator.aggregate(in_frame.text)
        if aggregation:
            out_frame = AggregatedTextFrame(
                text=aggregation.text,
                aggregated_by=aggregation.type,
            )
            out_frame.skip_tts = in_frame.skip_tts
            await self.push_frame(out_frame)

    async def _handle_llm_end(self, skip_tts: bool = False):
        # Flush any remaining aggregated text at the end of the LLM response
        aggregation = self._text_aggregator.text
        await self._text_aggregator.reset()
        text = aggregation.text.strip()
        if text:
            out_frame = AggregatedTextFrame(
                text=text,
                aggregated_by=aggregation.type,
            )
            out_frame.skip_tts = skip_tts
            await self.push_frame(out_frame)
