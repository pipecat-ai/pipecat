#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace):
    logger.info(f"Starting bot")

    # Create an HTTP session
    async with aiohttp.ClientSession() as session:
        tts = PiperTTSService(
            base_url=os.getenv("PIPER_BASE_URL"), aiohttp_session=session, sample_rate=24000
        )

        task = PipelineTask(Pipeline([tts, transport.output()]))

        # Register an event handler so we can play the audio when the client joins
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await task.queue_frames([TTSSpeakFrame(f"Hello there!"), EndFrame()])

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)


if __name__ == "__main__":
    from run import main

    main(run_example, transport_params=transport_params)
