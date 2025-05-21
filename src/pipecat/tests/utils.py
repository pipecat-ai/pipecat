#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    HeartbeatFrame,
    StartFrame,
    SystemFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class SleepFrame(SystemFrame):
    """This frame is used by test framework to introduce some sleep time before
    the next frame is pushed. This is useful to control system frames vs data or
    control frames.
    """

    sleep: float = 0.1


class HeartbeatsObserver(BaseObserver):
    def __init__(
        self,
        *,
        target: FrameProcessor,
        heartbeat_callback: Callable[[FrameProcessor, HeartbeatFrame], Awaitable[None]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._target = target
        self._callback = heartbeat_callback

    async def on_push_frame(self, data: FramePushed):
        src = data.source
        frame = data.frame

        if src == self._target and isinstance(frame, HeartbeatFrame):
            await self._callback(self._target, frame)


class QueuedFrameProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        queue: asyncio.Queue,
        queue_direction: FrameDirection,
        ignore_start: bool = True,
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
    expected_down_frames: Optional[Sequence[type]] = None,
    expected_up_frames: Optional[Sequence[type]] = None,
    ignore_start: bool = True,
    observers: Optional[List[BaseObserver]] = None,
    start_metadata: Optional[Dict[str, Any]] = None,
    send_end_frame: bool = True,
) -> Tuple[Sequence[Frame], Sequence[Frame]]:
    observers = observers or []
    start_metadata = start_metadata or {}

    received_up = asyncio.Queue()
    received_down = asyncio.Queue()
    source = QueuedFrameProcessor(
        queue=received_up,
        queue_direction=FrameDirection.UPSTREAM,
        ignore_start=ignore_start,
    )
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=ignore_start,
    )

    pipeline = Pipeline([source, processor, sink])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(start_metadata=start_metadata),
        observers=observers,
        cancel_on_idle_timeout=False,
    )

    async def push_frames():
        # Just give a little head start to the runner.
        await asyncio.sleep(0.01)
        for frame in frames_to_send:
            if isinstance(frame, SleepFrame):
                await asyncio.sleep(frame.sleep)
            else:
                await task.queue_frame(frame)

        if send_end_frame:
            await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), push_frames())

    #
    # Down frames
    #
    received_down_frames: Sequence[Frame] = []
    if expected_down_frames is not None:
        while not received_down.empty():
            frame = await received_down.get()
            if not isinstance(frame, EndFrame) or not send_end_frame:
                received_down_frames.append(frame)

        print("received DOWN frames =", received_down_frames)
        print("expected DOWN frames =", expected_down_frames)

        assert len(received_down_frames) == len(expected_down_frames)

        for real, expected in zip(received_down_frames, expected_down_frames):
            assert isinstance(real, expected)

    #
    # Up frames
    #
    received_up_frames: Sequence[Frame] = []
    if expected_up_frames is not None:
        while not received_up.empty():
            frame = await received_up.get()
            received_up_frames.append(frame)

        print("received UP frames =", received_up_frames)
        print("expected UP frames =", expected_up_frames)

        assert len(received_up_frames) == len(expected_up_frames)

        for real, expected in zip(received_up_frames, expected_up_frames):
            assert isinstance(real, expected)

    return (received_down_frames, received_up_frames)
