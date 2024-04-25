#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

from pipecat.frames.frames import AudioRawFrame, ImageRawFrame
from pipecat.processors.filter import Filter
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.transports.services.daily import DailyTransport, DailyParams

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url, token):
    transport = DailyTransport(
        room_url, token, "Test",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=True,
            camera_out_width=1280,
            camera_out_height=720
        )
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_video(participant["id"])

    # The ParallelPipeline is not really necessary here but it shows how you
    # would process audio and video concurrently in parallel pipelines.
    pipeline = Pipeline([transport.input(),
                         ParallelPipeline(
                             [Filter([AudioRawFrame])],
                             [Filter([ImageRawFrame])]),
                         transport.output()])

    runner = PipelineRunner()

    task = PipelineTask(pipeline)

    await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
