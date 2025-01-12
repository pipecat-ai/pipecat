#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List, Tuple

from loguru import logger

from pipecat.frames.frames import Frame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class GatedAggregator(FrameProcessor):
    """Accumulate frames, with custom functions to start and stop accumulation.
    Yields gate-opening frame before any accumulated frames, then ensuing frames
    until and not including the gate-closed frame.

    Doctest: FIXME to work with asyncio
    >>> from pipecat.frames.frames import ImageRawFrame

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         if isinstance(frame, TextFrame):
    ...             print(frame.text)
    ...         else:
    ...             print(frame.__class__.__name__)

    >>> aggregator = GatedAggregator(
    ...     gate_close_fn=lambda x: isinstance(x, LLMResponseStartFrame),
    ...     gate_open_fn=lambda x: isinstance(x, ImageRawFrame),
    ...     start_open=False)
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello")))
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello again.")))
    >>> asyncio.run(print_frames(aggregator, ImageRawFrame(image=bytes([]), size=(0, 0))))
    ImageRawFrame
    Hello
    Hello again.
    >>> asyncio.run(print_frames(aggregator, TextFrame("Goodbye.")))
    Goodbye.
    """

    def __init__(
        self,
        gate_open_fn,
        gate_close_fn,
        start_open,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        super().__init__()
        self._gate_open_fn = gate_open_fn
        self._gate_close_fn = gate_close_fn
        self._gate_open = start_open
        self._direction = direction
        self._accumulator: List[Tuple[Frame, FrameDirection]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        # Ignore frames that are not following the direction of this gate.
        if direction != self._direction:
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
            for f, d in self._accumulator:
                await self.push_frame(f, d)
            self._accumulator = []
        else:
            self._accumulator.append((frame, direction))
