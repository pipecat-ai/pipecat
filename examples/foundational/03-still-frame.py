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
from pipecat.services.fal import FalImageGenService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv

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

        imagegen = FalImageGenService(
            params=FalImageGenService.InputParams(image_size="square_hd"),
            aiohttp_session=session,
            key=os.getenv("FAL_KEY"),
        )

        runner = PipelineRunner()

        task = PipelineTask(Pipeline([imagegen, transport.output()]))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frame(TextFrame("a cat in the style of picasso"))

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
