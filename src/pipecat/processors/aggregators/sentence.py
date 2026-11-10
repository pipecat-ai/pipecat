#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Text sentence aggregation processor for Pipecat.

This module provides a frame processor that accumulates text frames into
complete sentences, only outputting when a sentence-ending pattern is detected.
"""

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import match_endofsentence


class SentenceAggregator(FrameProcessor):
    """Aggregates text frames into complete sentences.

    This processor accumulates incoming text frames until a sentence-ending
    pattern is detected, then outputs the complete sentence as a single frame.
    Useful for ensuring downstream processors receive coherent, complete sentences
    rather than fragmented text.

    A frame that holds the end of one sentence and the start of the next emits
    only the completed sentence. The rest stays buffered for the following one.

    Frame input/output::

        TextFrame("Hello,") -> None
        TextFrame(" world.") -> TextFrame("Hello, world.")
        TextFrame("Hello. How are") -> TextFrame("Hello.")
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

        # We ignore interim and eager (provisional) transcriptions at this point.
        if isinstance(frame, (InterimTranscriptionFrame, EagerTranscriptionFrame)):
            return

        if isinstance(frame, TextFrame):
            self._aggregation += frame.text
            # ``match_endofsentence`` returns the end of the first sentence.
            while self._aggregation:
                end = match_endofsentence(self._aggregation)
                if not end:
                    break
                # Keep a coarse chunk's post-boundary text together for the next
                # sentence. When only trailing whitespace follows the boundary,
                # preserve the existing eager, word-by-word output.
                remainder = self._aggregation[end:]
                split = len(self._aggregation) if remainder and not remainder.strip() else end
                sentence = self._aggregation[:split]
                self._aggregation = self._aggregation[split:]
                if sentence:
                    await self.push_frame(TextFrame(sentence))
        elif isinstance(frame, InterruptionFrame):
            self._aggregation = ""
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            if self._aggregation:
                await self.push_frame(TextFrame(self._aggregation))
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)
