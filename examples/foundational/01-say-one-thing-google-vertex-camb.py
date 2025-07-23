#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.google_vertex_camb.tts import GoogleVertexCambTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv()


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    # Check required environment variables
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    endpoint_id = os.getenv("ENDPOINT_ID")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not endpoint_id:
        raise ValueError("ENDPOINT_ID environment variable is required")
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")

    # Configure Google Vertex CAMB TTS with voice cloning parameters
    params = GoogleVertexCambTTSService.InputParams(
        reference_audio_path=os.getenv("REFERENCE_AUDIO_PATH"),  # Optional: path to reference audio for voice cloning
        reference_text=os.getenv("REFERENCE_TEXT"),  # Optional: transcription of reference audio
        language="en-us"  # Language for synthesis
    )

    tts = GoogleVertexCambTTSService(
        project_id=project_id,
        location=location,
        endpoint_id=endpoint_id,
        credentials_path=credentials_path,
        params=params,
        sample_rate=44100
    )

    task = PipelineTask(Pipeline([tts, transport.output()]))

    # Register an event handler so we can play the audio when the client joins
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected, queuing TTS frame")
        await task.queue_frames([
            TTSSpeakFrame("Hello there! I'm using Google Vertex AI with CAMB's MARS7 model for high-quality speech synthesis."),
            EndFrame()
        ])

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
