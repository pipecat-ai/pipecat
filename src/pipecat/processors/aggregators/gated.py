#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List

from pipecat.frames.frames import Frame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from loguru import logger


class GatedAggregator(FrameProcessor):
    """Accumulate frames, with custom functions to start and stop accumulation.
    Yields gate-opening frame before any accumulated frames, then ensuing frames
    until and not including the gate-closed frame.

    >>> from pipecat.pipeline.frames import ImageFrame

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         if isinstance(frame, TextFrame):
    ...             print(frame.text)
    ...         else:
    ...             print(frame.__class__.__name__)

    >>> aggregator = GatedAggregator(
    ...     gate_close_fn=lambda x: isinstance(x, LLMResponseStartFrame),
    ...     gate_open_fn=lambda x: isinstance(x, ImageFrame),
    ...     start_open=False)
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello")))
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello again.")))
    >>> asyncio.run(print_frames(aggregator, ImageFrame(image=bytes([]), size=(0, 0))))
    ImageFrame
    Hello
    Hello again.
    >>> asyncio.run(print_frames(aggregator, TextFrame("Goodbye.")))
    Goodbye.
    """

    def __init__(self, gate_open_fn, gate_close_fn, start_open):
        super().__init__()
        self._gate_open_fn = gate_open_fn
        self._gate_close_fn = gate_close_fn
        self._gate_open = start_open
        self._accumulator: List[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        old_state = self._gate_open
        if self._gate_open:
            self._gate_open = not self._gate_close_fn(frame)
        else:
            self._gate_open = self._gate_open_fn(frame)

        if old_state != self._gate_open:
            state = "open" if self._gate_open else "closed"
            logger.debug(f"Gate is now {state} because of {frame}")

        if self._gate_open:
            await self.push_frame(frame, direction)
            for frame in self._accumulator:
                await self.push_frame(frame, direction)
            self._accumulator = []
        else:
            self._accumulator.append(frame)
