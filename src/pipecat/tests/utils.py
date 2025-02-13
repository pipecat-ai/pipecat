#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Any, Awaitable, Callable, Dict, Sequence, Tuple

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    HeartbeatFrame,
    StartFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


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
    def __init__(
        self, queue: asyncio.Queue, queue_direction: FrameDirection, ignore_start: bool = True
    ):
        super().__init__()
        self._queue = queue
        self._queue_direction = queue_direction
        self._ignore_start = ignore_start

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == self._queue_direction:
            if not isinstance(frame, StartFrame) or not self._ignore_start:
                await self._queue.put(frame)
        await self.push_frame(frame, direction)


async def run_test(
    processor: FrameProcessor,
    *,
    frames_to_send: Sequence[Frame],
    expected_down_frames: Sequence[type],
    expected_up_frames: Sequence[type] = [],
    ignore_start: bool = True,
    start_metadata: Dict[str, Any] = {},
    send_end_frame: bool = True,
) -> Tuple[Sequence[Frame], Sequence[Frame]]:
    received_up = asyncio.Queue()
    received_down = asyncio.Queue()
    source = QueuedFrameProcessor(received_up, FrameDirection.UPSTREAM, ignore_start)
    sink = QueuedFrameProcessor(received_down, FrameDirection.DOWNSTREAM, ignore_start)

    pipeline = Pipeline([source, processor, sink])

    task = PipelineTask(pipeline, params=PipelineParams(start_metadata=start_metadata))

    for frame in frames_to_send:
        await task.queue_frame(frame)

    if send_end_frame:
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await runner.run(task)

    #
    # Down frames
    #
    received_down_frames: Sequence[Frame] = []
    while not received_down.empty():
        frame = await received_down.get()
        if not isinstance(frame, EndFrame) or not send_end_frame:
            received_down_frames.append(frame)

    print("received DOWN frames =", received_down_frames)

    assert len(received_down_frames) == len(expected_down_frames)

    for real, expected in zip(received_down_frames, expected_down_frames):
        assert isinstance(real, expected)

    #
    # Up frames
    #
    received_up_frames: Sequence[Frame] = []
    while not received_up.empty():
        frame = await received_up.get()
        received_up_frames.append(frame)

    print("received UP frames =", received_up_frames)

    assert len(received_up_frames) == len(expected_up_frames)

    for real, expected in zip(received_up_frames, expected_up_frames):
        assert isinstance(real, expected)

    return (received_down_frames, received_up_frames)
