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

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google.image import GoogleImageGenService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Show a still frame image",
            DailyParams(camera_out_enabled=True, camera_out_width=1024, camera_out_height=1024),
        )

        imagegen = GoogleImageGenService(
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        runner = PipelineRunner()

        task = PipelineTask(
            Pipeline([imagegen, transport.output()]),
            params=PipelineParams(enable_metrics=True),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frame(TextFrame("a cat in the style of picasso"))
            await task.queue_frame(TextFrame("a dog in the style of picasso"))
            await task.queue_frame(TextFrame("a fish in the style of picasso"))

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
