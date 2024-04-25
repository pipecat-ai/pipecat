#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from pipecat.frames.frames import TextFrame
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


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            None,
            "Show a still frame image",
            DailyParams(
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1024
            )
        )

        imagegen = FalImageGenService(
            params=FalImageGenService.InputParams(
                image_size="square_hd"
            ),
            aiohttp_session=session,
            key=os.getenv("FAL_KEY"),
        )

        runner = PipelineRunner()

        task = PipelineTask(Pipeline([imagegen, transport.output()]))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Note that we do not put an EndFrame() item in the pipeline for this demo.
            # This means that the bot will stay in the channel until it times out.
            # An EndFrame() in the pipeline would cause the transport to shut
            # down.
            await task.queue_frames([TextFrame("a cat in the style of picasso")])

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
