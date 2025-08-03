#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

# /// script
# dependencies = [
#   "pipecat-ai[daily,webrtc,runner,gstreamer]>=0.0.77",
# ]
# ///

from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.gstreamer.pipeline_source import GStreamerPipelineSource
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
    ),
    "webrtc": lambda: TransportParams(
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
    ),
}


async def run_bot(transport: BaseTransport):
    logger.info(f"Starting bot with video test source")

    gst = GStreamerPipelineSource(
        pipeline='videotestsrc ! capsfilter caps="video/x-raw,width=1280,height=720,framerate=30/1"',
        out_params=GStreamerPipelineSource.OutputParams(
            video_width=1280, video_height=720, clock_sync=False
        ),
    )

    pipeline = Pipeline(
        [
            gst,  # GStreamer test source
            transport.output(),  # Transport bot output
        ]
    )

    task = PipelineTask(pipeline)

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
