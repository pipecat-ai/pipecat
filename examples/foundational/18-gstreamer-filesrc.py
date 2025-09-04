#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse

from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.gstreamer.pipeline_source import GStreamerPipelineSource
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Pipecat Video Streaming Bot")
parser.add_argument("-i", "--input", type=str, required=True, help="Input video file")
args = parser.parse_args()

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
    ),
    "webrtc": lambda: TransportParams(
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot with video input: {args.input}")

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

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
