#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Stateless text transformation processor for Pipecat."""

from typing import Callable, Coroutine, Union

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class StatelessTextTransformer(FrameProcessor):
    """Processor that applies transformation functions to text frames.

    This processor intercepts TextFrame objects and applies a user-provided
    transformation function to the text content. The function can be either
    synchronous or asynchronous (coroutine).
    """

    def __init__(
        self, transform_fn: Union[Callable[[str], str], Callable[[str], Coroutine[None, None, str]]]
    ):
        """Initialize the text transformer.

        Args:
            transform_fn: Function to apply to text content. Can be synchronous
                (str -> str) or asynchronous (str -> Coroutine[None, None, str]).
        """
        super().__init__()
        self._transform_fn = transform_fn

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, applying transformation to text frames.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            result = self._transform_fn(frame.text)
            if isinstance(result, Coroutine):
                result = await result
            await self.push_frame(TextFrame(text=result))
        else:
            await self.push_frame(frame, direction)
