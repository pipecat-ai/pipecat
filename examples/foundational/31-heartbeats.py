#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class NullProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)


async def main():
    """This test shows heartbeat monitoring.

    A warning is dispalyed when heartbeats are not received within the
    default (5 seconds) timeout.
    """
    pipeline = Pipeline([NullProcessor()])

    task = PipelineTask(pipeline, params=PipelineParams(enable_heartbeats=True))

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
