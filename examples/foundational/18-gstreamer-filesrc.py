#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure_with_args

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.gstreamer.pipeline_source import GStreamerPipelineSource
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
        parser.add_argument("-i", "--input", type=str, required=True, help="Input video file")

        (room_url, _, args) = await configure_with_args(session, parser)

        transport = DailyTransport(
            room_url,
            None,
            "GStreamer",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1280,
                camera_out_height=720,
                camera_out_is_live=True,
            ),
        )

        gst = GStreamerPipelineSource(
            pipeline=f"filesrc location={args.input}",
            out_params=GStreamerPipelineSource.OutputParams(
                video_width=1280,
                video_height=720,
            ),
        )

        pipeline = Pipeline(
            [
                gst,  # GStreamer file source
                transport.output(),  # Transport bot output
            ]
        )

        task = PipelineTask(pipeline)

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
