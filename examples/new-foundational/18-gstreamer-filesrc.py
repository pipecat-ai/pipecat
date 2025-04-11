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
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


# Parse command line arguments
# This will be used to pass the input video file to the bot
# You can run the bot with a command like:
# python 18-gstreamer-filesrc.py -i path/to/video.mp4
def parse_arguments():
    parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video file")
    return parser.parse_args()


args = parse_arguments()


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot with video input: {args.input}")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            camera_out_enabled=True,
            camera_out_is_live=True,
            camera_out_width=1280,
            camera_out_height=720,
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

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
