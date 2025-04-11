#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.fal.image import FalImageGenService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot")

    # Create a transport using the WebRTC connection
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            camera_out_enabled=True,
            camera_out_width=1024,
            camera_out_height=1024,
        ),
    )

    # Create an HTTP session
    async with aiohttp.ClientSession() as session:
        imagegen = FalImageGenService(
            params=FalImageGenService.InputParams(image_size="square_hd"),
            aiohttp_session=session,
            key=os.getenv("FAL_KEY"),
        )

        task = PipelineTask(Pipeline([imagegen, transport.output()]))

        # Register an event handler so we can play the audio when the client joins
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await task.queue_frame(TextFrame("a cat in the style of picasso"))

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")

        @transport.event_handler("on_client_closed")
        async def on_client_closed(transport, client):
            logger.info(f"Client closed connection")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
