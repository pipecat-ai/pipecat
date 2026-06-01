#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    pipeline = Pipeline([tts, transport.output()])

    worker = PipelineWorker(pipeline)

    async def say_something():
        await asyncio.sleep(1)
        await worker.queue_frames([TTSSpeakFrame("Hello there, how is it going!"), EndFrame()])

    runner = WorkerRunner(handle_sigint=False if sys.platform == "win32" else True)

    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), say_something())


if __name__ == "__main__":
    asyncio.run(main())
