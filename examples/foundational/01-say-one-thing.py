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
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url, None, "Say One Thing", DailyParams(audio_out_enabled=True))

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        runner = PipelineRunner()

        task = PipelineTask(Pipeline([tts, transport.output()]))

        # Register an event handler so we can play the audio when the
        # participant joins.
        @transport.event_handler("on_participant_joined")
        async def on_new_participant_joined(transport, participant):
            participant_name = participant["info"]["userName"] or ''
            await task.queue_frames([TextFrame(f"Hello there, {participant_name}!"), EndFrame()])

        await runner.run(task)

if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
