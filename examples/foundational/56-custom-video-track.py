#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example demonstrating custom video tracks output with Daily transport.

This example outputs two video track simultaneously:
  - The default camera track with an animated color gradient pattern.
  - A custom "blue" track with the same pattern but with a blue tint applied.

The pattern generator pushes frames to the default camera. A second processor
(BlueTintProcessor) duplicates each frame, applies a blue tint, and pushes it
to the "blue" custom video destination.

Run with: python examples/foundational/56-custom-video-track.py -t daily
"""

import asyncio
import math
import time

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    StartFrame,
    SystemFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyCustomVideoTrackParams, DailyParams

WIDTH = 320
HEIGHT = 240
FPS = 30

transport_params = {
    "daily": lambda: DailyParams(
        video_out_enabled=True,
        video_out_width=WIDTH,
        video_out_height=HEIGHT,
        video_out_framerate=FPS,
        video_out_destinations=["blue"],
        custom_video_track_params={
            "blue": DailyCustomVideoTrackParams(
                width=WIDTH,
                height=HEIGHT,
                send_settings={
                    "maxQuality": "low",
                    "encodings": {
                        "low": {
                            "maxBitrate": 500_000,
                            "maxFramerate": FPS,
                        }
                    },
                },
            ),
        },
    ),
}


def generate_gradient_frame(width: int, height: int, t: float) -> np.ndarray:
    """Generate an animated gradient pattern.

    Creates a smooth color gradient that shifts over time using sine waves
    for each RGB channel at different frequencies.
    """
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)

    r = ((np.sin(2 * math.pi * (xv + t * 0.3)) + 1) / 2 * 255).astype(np.uint8)
    g = ((np.sin(2 * math.pi * (yv + t * 0.5)) + 1) / 2 * 255).astype(np.uint8)
    b = ((np.sin(2 * math.pi * (xv + yv + t * 0.7)) + 1) / 2 * 255).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


class VideoPatternGenerator(FrameProcessor):
    """Generates an animated gradient pattern and pushes it as video frames."""

    def __init__(self, width: int, height: int, fps: int):
        super().__init__()
        self._width = width
        self._height = height
        self._fps = fps
        self._generate_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._start()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _start(self):
        self._generate_task = self.create_task(self._generate_loop(), "video_generate_loop")

    async def _stop(self):
        if self._generate_task:
            await self.cancel_task(self._generate_task)
            self._generate_task = None

    async def _generate_loop(self):
        interval = 1.0 / self._fps
        start = time.monotonic()

        while True:
            t = time.monotonic() - start

            pattern = generate_gradient_frame(self._width, self._height, t)

            frame = OutputImageRawFrame(
                image=pattern.tobytes(),
                size=(self._width, self._height),
                format="RGB",
            )
            await self.push_frame(frame)

            elapsed = time.monotonic() - start - t
            await asyncio.sleep(max(0, interval - elapsed))


class BlueTintProcessor(FrameProcessor):
    """Duplicates OutputImageRawFrames with a blue tint for a custom video destination."""

    def __init__(self, destination: str):
        super().__init__()
        self._destination = destination

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputImageRawFrame):
            # Pass through the original frame.
            await self.push_frame(frame, direction)

            # Create a blue-tinted copy for the custom destination.
            img = np.frombuffer(frame.image, dtype=np.uint8).reshape(
                (frame.size[1], frame.size[0], 3)
            )
            tinted = img.copy()
            tinted[:, :, 0] = (tinted[:, :, 0] * 0.3).astype(np.uint8)  # R
            tinted[:, :, 1] = (tinted[:, :, 1] * 0.3).astype(np.uint8)  # G
            tinted[:, :, 2] = np.clip(tinted[:, :, 2].astype(np.uint16) + 80, 0, 255).astype(
                np.uint8
            )  # B

            blue_frame = OutputImageRawFrame(
                image=tinted.tobytes(),
                size=frame.size,
                format=frame.format,
            )
            blue_frame.transport_destination = self._destination
            await self.push_frame(blue_frame)
        else:
            await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting dual video track bot")

    generator = VideoPatternGenerator(WIDTH, HEIGHT, FPS)
    blue_tint = BlueTintProcessor(destination="blue")

    task = PipelineTask(
        Pipeline([generator, blue_tint, transport.output()]),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.queue_frame(EndFrame())

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
