#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from typing import List, Tuple

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ControlFrame,
    Frame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class EndTestFrame(ControlFrame):
    pass


class QueuedFrameProcessor(FrameProcessor):
    def __init__(self, queue: asyncio.Queue, ignore_start: bool = True):
        super().__init__()
        self._queue = queue
        self._ignore_start = ignore_start

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._ignore_start and isinstance(frame, StartFrame):
            return
        await self._queue.put(frame)


async def run_test(
    processor: FrameProcessor,
    frames_to_send: List[Frame],
    expected_down_frames: List[type],
    expected_up_frames: List[type] = [],
) -> Tuple[List[Frame], List[Frame]]:
    received_up = asyncio.Queue()
    received_down = asyncio.Queue()
    up_processor = QueuedFrameProcessor(received_up)
    down_processor = QueuedFrameProcessor(received_down)

    up_processor.link(processor)
    processor.link(down_processor)

    await processor.queue_frame(StartFrame(clock=SystemClock()))

    for frame in frames_to_send:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    await processor.queue_frame(EndTestFrame())
    await processor.queue_frame(EndTestFrame(), FrameDirection.UPSTREAM)

    #
    # Down frames
    #
    received_down_frames: List[Frame] = []
    running = True
    while running:
        frame = await received_down.get()
        running = not isinstance(frame, EndTestFrame)
        if running:
            received_down_frames.append(frame)

    print("received DOWN frames =", received_down_frames)

    assert len(received_down_frames) == len(expected_down_frames)

    for real, expected in zip(received_down_frames, expected_down_frames):
        assert isinstance(real, expected)

    #
    # Up frames
    #
    received_up_frames: List[Frame] = []
    running = True
    while running:
        frame = await received_up.get()
        running = not isinstance(frame, EndTestFrame)
        if running:
            received_up_frames.append(frame)

    print("received UP frames =", received_up_frames)

    assert len(received_up_frames) == len(expected_up_frames)

    for real, expected in zip(received_up_frames, expected_up_frames):
        assert isinstance(real, expected)

    return (received_down_frames, received_up_frames)
