#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Coroutine

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class StatelessTextTransformer(FrameProcessor):
    """This processor calls the given function on any text in a text frame.

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         print(frame.text)

    >>> aggregator = StatelessTextTransformer(lambda x: x.upper())
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello")))
    HELLO
    """

    def __init__(self, transform_fn):
        super().__init__()
        self._transform_fn = transform_fn

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            result = self._transform_fn(frame.text)
            if isinstance(result, Coroutine):
                result = await result
            await self.push_frame(TextFrame(text=result))
        else:
            await self.push_frame(frame, direction)
