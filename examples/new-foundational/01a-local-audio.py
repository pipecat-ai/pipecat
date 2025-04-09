#
# Copyright (c) 2024–2025, Daily
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
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    pipeline = Pipeline([tts, transport.output()])

    task = PipelineTask(pipeline)

    async def say_something():
        await asyncio.sleep(1)
        await task.queue_frames([TTSSpeakFrame("Hello there, how is it going!"), EndFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    await asyncio.gather(runner.run(task), say_something())


if __name__ == "__main__":
    asyncio.run(main())
