#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

from loguru import logger

from pipecat.frames.frames import Frame, SystemFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class NullProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # Only pass system frames (e.g. StartFrame, CancelFrame) so heartbeat
        # frames are swallowed, simulating a stalled pipeline.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)


async def main():
    """This example shows heartbeat monitoring.

    A warning is displayed when heartbeats are not received within the
    configured timeout, and the on_heartbeat_timeout event handler is invoked.
    """
    pipeline = Pipeline([NullProcessor()])

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_heartbeats=True))

    @worker.event_handler("on_heartbeat_timeout")
    async def on_heartbeat_timeout(worker: PipelineWorker):
        logger.warning("Heartbeat timeout detected — pipeline may be stalled")

    runner = WorkerRunner()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
