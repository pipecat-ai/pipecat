#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Testing utilities for Pipecat pipeline components."""

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
    """A system frame that introduces a sleep delay in the test pipeline.

    This frame is used by the test framework to control timing between
    frame processing, allowing tests to separate system frames from
    data or control frames.

    Parameters:
        sleep: Duration to sleep in seconds before processing the next frame.
    """

    sleep: float = 0.2


class HeartbeatsObserver(BaseObserver):
    """Observer that monitors heartbeat frames from a specific processor.

    This observer watches for HeartbeatFrames from a target processor and
    invokes a callback when they are detected, useful for testing timing
    and lifecycle events.
    """

    def __init__(
        self,
        *,
        target: FrameProcessor,
        heartbeat_callback: Callable[[FrameProcessor, HeartbeatFrame], Awaitable[None]],
        **kwargs,
    ):
        """Initialize the heartbeats observer.

        Args:
            target: The frame processor to monitor for heartbeat frames.
            heartbeat_callback: Async callback function to invoke when heartbeats are detected.
            **kwargs: Additional arguments passed to the parent observer.
        """
        super().__init__(**kwargs)
        self._target = target
        self._callback = heartbeat_callback

    async def on_push_frame(self, data: FramePushed):
        """Handle frame push events and detect heartbeats from target processor.

        Args:
            data: The frame push event data containing source and frame information.
        """
        src = data.source
        frame = data.frame

        if src == self._target and isinstance(frame, HeartbeatFrame):
            await self._callback(self._target, frame)


class QueuedFrameProcessor(FrameProcessor):
    """A processor that captures frames in a queue for testing purposes.

    This processor intercepts frames flowing in a specific direction and
    stores them in a queue for later inspection during testing, while
    still allowing the frames to continue through the pipeline.
    """

    def __init__(
        self,
        *,
        queue: asyncio.Queue,
        queue_direction: FrameDirection,
        ignore_start: bool = True,
    ):
        """Initialize the queued frame processor.

        Args:
            queue: The asyncio queue to store captured frames.
            queue_direction: The direction of frames to capture (UPSTREAM or DOWNSTREAM).
            ignore_start: Whether to ignore StartFrames when capturing.
        """
        super().__init__(enable_direct_mode=True)
        self._queue = queue
        self._queue_direction = queue_direction
        self._ignore_start = ignore_start

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and capture them in the queue if they match the direction.

        Args:
            frame: The frame to process.
            direction: The direction the frame is flowing.
        """
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
    pipeline_params: Optional[PipelineParams] = None,
    send_end_frame: bool = True,
) -> Tuple[Sequence[Frame], Sequence[Frame]]:
    """Run a test pipeline with the specified processor and validate frame flow.

    This function creates a test pipeline with the given processor, sends the
    specified frames through it, and validates that the expected frames are
    received in both upstream and downstream directions.

    Args:
        processor: The frame processor to test.
        frames_to_send: Sequence of frames to send through the processor.
        expected_down_frames: Expected frame types flowing downstream (optional).
        expected_up_frames: Expected frame types flowing upstream (optional).
        ignore_start: Whether to ignore StartFrames in frame validation.
        observers: Optional list of observers to attach to the pipeline.
        pipeline_params: Optional pipeline parameters.
        send_end_frame: Whether to send an EndFrame at the end of the test.

    Returns:
        Tuple containing (downstream_frames, upstream_frames) that were received.

    Raises:
        AssertionError: If the received frames don't match the expected frame types.
    """
    observers = observers or []
    pipeline_params = pipeline_params or PipelineParams()

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
        params=pipeline_params,
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
