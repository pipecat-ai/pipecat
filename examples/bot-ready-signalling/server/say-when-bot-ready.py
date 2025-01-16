#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url, None, "Say One Thing", DailyParams(audio_out_enabled=True)
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        runner = PipelineRunner()

        task = PipelineTask(Pipeline([tts, transport.output()]))

        # Register an event handler so we can play the audio when we receive a specific message
        @transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            logger.debug(f"Received app message: {message} - {sender}")
            if "message" not in message:
                return
            await task.queue_frames(
                [TTSSpeakFrame(f"Hello there, how are you doing today ?"), EndFrame()]
            )

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
