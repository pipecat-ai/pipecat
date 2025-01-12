#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import EndFrame, Frame, InterimTranscriptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import match_endofsentence


class SentenceAggregator(FrameProcessor):
    """This frame processor aggregates text frames into complete sentences.

    Frame input/output:
        TextFrame("Hello,") -> None
        TextFrame(" world.") -> TextFrame("Hello world.")

    Doctest: FIXME to work with asyncio
    >>> import asyncio
    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         print(frame.text)

    >>> aggregator = SentenceAggregator()
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello,")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" world.")))
    Hello, world.
    """

    def __init__(self):
        super().__init__()
        self._aggregation = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
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
