#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputImageRawFrame,
    OutputImageRawFrame,
    TextFrame,
    TTSTextFrame,
    UserImageRequestFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver, FrameEndpoint
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
)
from pipecat.processors.gstreamer.pipeline_source import GStreamerPipelineSource
from pipecat.services.moondream.vision import MoondreamService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


class AlertProcessor(FrameProcessor):
    def __init__(self, connection: SmallWebRTCConnection):
        super().__init__()
        self._connection = connection

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            text = frame.text.strip().upper()
            message_frame = RTVIServerMessageFrame(data=text)
            await self.push_frame(message_frame)

        await self.push_frame(frame, direction)


class UserImageRequester(FrameProcessor):
    def __init__(self, participant_id: Optional[str] = None):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputImageRawFrame):
            await self.push_frame(frame)
            # logger.info(f"UserImageRequester received image frame with size: {frame.size}")
            text_frame = TextFrame(
                "Are there people in the bottom right corner of the image? Only answer with YES or NO."
            )
            await self.push_frame(text_frame)
            input_frame = InputImageRawFrame(
                image=frame.image,
                size=frame.size,
                format=frame.format,
            )
            await self.push_frame(input_frame)
        else:
            await self.push_frame(frame, direction)


async def run_bot(webrtc_connection: SmallWebRTCConnection, args: argparse.Namespace):
    logger.info(f"Starting bot with video input: {args.input}")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
        ),
    )

    gst = GStreamerPipelineSource(
        pipeline=(f"rtspsrc location={args.input} ! decodebin ! autovideosink"),
        out_params=GStreamerPipelineSource.OutputParams(
            video_width=1280,
            video_height=720,
        ),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # If you run into weird description, try with use_cpu=True
    moondream = MoondreamService()

    ir = UserImageRequester()
    va = VisionImageFrameAggregator()
    alert = AlertProcessor(connection=webrtc_connection)

    pipeline = Pipeline(
        [
            gst,  # GStreamer file source
            rtvi,
            ir,
            # debug,
            va,
            moondream,
            alert,  # Send an email alert or something if the door is open
            transport.output(),  # Transport bot output
        ]
    )

    task = PipelineTask(
        pipeline,
        observers=[
            RTVIObserver(rtvi),
            DebugLogObserver(
                frame_types={
                    # TextFrame: None,
                    TextFrame: (MoondreamService, FrameEndpoint.SOURCE),
                    # InputImageRawFrame: None,
                    EndFrame: None,
                }
            ),
        ],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info(f"Bot ready: {rtvi}")
        await rtvi.set_bot_ready()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


if __name__ == "__main__":
    from run import main

    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video file")

    main(parser)
