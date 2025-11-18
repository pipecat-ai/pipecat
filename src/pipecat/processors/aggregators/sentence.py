#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Text sentence aggregation processor for Pipecat.

This module provides a frame processor that accumulates text frames into
complete sentences, only outputting when a sentence-ending pattern is detected.
"""

from pipecat.frames.frames import EndFrame, Frame, InterimTranscriptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import match_endofsentence


class SentenceAggregator(FrameProcessor):
    """Aggregates text frames into complete sentences.

    This processor accumulates incoming text frames until a sentence-ending
    pattern is detected, then outputs the complete sentence as a single frame.
    Useful for ensuring downstream processors receive coherent, complete sentences
    rather than fragmented text.

    Frame input/output::

        TextFrame("Hello,") -> None
        TextFrame(" world.") -> TextFrame("Hello, world.")
    """

    def __init__(self):
        """Initialize the sentence aggregator.

        Sets up internal state for accumulating text frames into complete sentences.
        """
        super().__init__()
        self._aggregation = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and aggregate text into complete sentences.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # We ignore interim description at this point.
        if isinstance(frame, InterimTranscriptionFrame):
            return

        if isinstance(frame, TextFrame):
            self._aggregation += frame.text
            if match_endofsentence(self._aggregation):
                await self.push_frame(TextFrame(self._aggregation))
                self._aggregation = ""
        elif isinstance(frame, EndFrame):
            if self._aggregation:
                await self.push_frame(TextFrame(self._aggregation))
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)
