#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vision image frame aggregation for Pipecat.

This module provides frame aggregation functionality to combine text and image
frames into vision frames for multimodal processing.
"""

from pipecat.frames.frames import Frame, InputImageRawFrame, TextFrame, VisionImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class VisionImageFrameAggregator(FrameProcessor):
    """Aggregates consecutive text and image frames into vision frames.

    This aggregator waits for a consecutive TextFrame and an InputImageRawFrame.
    After the InputImageRawFrame arrives it will output a VisionImageRawFrame
    combining both the text and image data for multimodal processing.

    >>> from pipecat.frames.frames import ImageFrame

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         print(frame)

    >>> aggregator = VisionImageFrameAggregator()
    >>> asyncio.run(print_frames(aggregator, TextFrame("What do you see?")))
    >>> asyncio.run(print_frames(aggregator, ImageFrame(image=bytes([]), size=(0, 0))))
    VisionImageFrame, text: What do you see?, image size: 0x0, buffer size: 0 B
    """

    def __init__(self):
        """Initialize the vision image frame aggregator.

        The aggregator starts with no cached text, waiting for the first
        TextFrame to arrive before it can create vision frames.
        """
        super().__init__()
        self._describe_text = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and aggregate text with images.

        Caches TextFrames and combines them with subsequent InputImageRawFrames
        to create VisionImageRawFrames. Other frames are passed through unchanged.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self._describe_text = frame.text
        elif isinstance(frame, InputImageRawFrame):
            if self._describe_text:
                frame = VisionImageRawFrame(
                    text=self._describe_text,
                    image=frame.image,
                    size=frame.size,
                    format=frame.format,
                )
                await self.push_frame(frame)
                self._describe_text = None
        else:
            await self.push_frame(frame, direction)
