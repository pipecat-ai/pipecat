#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import LocalAudioTransport

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        transport = LocalAudioTransport(TransportParams(audio_out_enabled=True))

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        pipeline = Pipeline([tts, transport.output()])

        task = PipelineTask(pipeline)

        async def say_something():
            await asyncio.sleep(1)
            await task.queue_frames([TextFrame("Hello there!"), EndFrame()])

        runner = PipelineRunner()

        await asyncio.gather(runner.run(task), say_something())


if __name__ == "__main__":
    asyncio.run(main())
