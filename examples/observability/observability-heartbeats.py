#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

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

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_heartbeats=True))

    runner = WorkerRunner()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
