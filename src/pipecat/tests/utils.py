#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence, Tuple

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ControlFrame,
    Frame,
    HeartbeatFrame,
    StartFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio import TaskManager


@dataclass
class EndTestFrame(ControlFrame):
    pass


class HeartbeatsObserver(BaseObserver):
    def __init__(
        self,
        *,
        target: FrameProcessor,
        heartbeat_callback: Callable[[FrameProcessor, HeartbeatFrame], Awaitable[None]],
    ):
        self._target = target
        self._callback = heartbeat_callback

    async def on_push_frame(
        self,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
        timestamp: int,
    ):
        if src == self._target and isinstance(frame, HeartbeatFrame):
            await self._callback(self._target, frame)


class QueuedFrameProcessor(FrameProcessor):
    def __init__(self, queue: asyncio.Queue, ignore_start: bool = True):
        super().__init__()
        self._queue = queue
        self._ignore_start = ignore_start

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._ignore_start and isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
        else:
            await self._queue.put(frame)
            await self.push_frame(frame, direction)


async def run_test(
    processor: FrameProcessor,
    frames_to_send: Sequence[Frame],
    expected_down_frames: Sequence[type],
    expected_up_frames: Sequence[type] = [],
) -> Tuple[Sequence[Frame], Sequence[Frame]]:
    received_up = asyncio.Queue()
    received_down = asyncio.Queue()
    source = QueuedFrameProcessor(received_up)
    sink = QueuedFrameProcessor(received_down)

    source.link(processor)
    processor.link(sink)

    task_manager = TaskManager()
    task_manager.set_event_loop(asyncio.get_event_loop())
    await source.queue_frame(StartFrame(clock=SystemClock(), task_manager=task_manager))

    for frame in frames_to_send:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    await processor.queue_frame(EndTestFrame())
    await processor.queue_frame(EndTestFrame(), FrameDirection.UPSTREAM)

    #
    # Down frames
    #
    received_down_frames: Sequence[Frame] = []
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
    received_up_frames: Sequence[Frame] = []
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
