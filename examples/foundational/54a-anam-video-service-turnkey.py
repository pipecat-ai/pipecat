#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import logging
import os

import aiohttp
from anam import PersonaConfig
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anam.video import AnamVideoService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyTransport

# Configure Python standard logging to show INFO messages from anam SDK
logging.getLogger("anam").setLevel(logging.DEBUG)

load_dotenv(override=True)

REQUIRED_ENV_VARS = ["ANAM_API_KEY", "ANAM_AVATAR_ID", "ANAM_LLM_ID", "ANAM_VOICE_ID"]
missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

ANAM_SAMPLE_RATE = 24000

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=720,
        video_out_height=480,
        audio_out_sample_rate=48000,  # Anam WebRTC output (OPUS 48kHz stereo)
        audio_out_channels=2,
        audio_in_sample_rate=16000,  # WebRTC input
        audio_in_channels=1,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    avatar_id = os.getenv("ANAM_AVATAR_ID").strip().strip('"')
    persona_config = PersonaConfig(
        avatar_id=avatar_id,
        voice_id=os.getenv("ANAM_VOICE_ID").strip().strip('"'),
        llm_id=os.getenv("ANAM_LLM_ID").strip().strip('"'),
        enable_audio_passthrough=False,
    )
    logger.info(f"Persona config: {persona_config}")

    anam = AnamVideoService(
        api_key=os.getenv("ANAM_API_KEY"),
        enable_turnkey=True,
        persona_config=persona_config,
        api_base_url="https://api.anam.ai",
        api_version="v1",
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            anam,  # Video Avatar (sends user audio to Anam and returns synchronised audio/video)
            transport.output(),  # Transport bot output
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Updating publishing settings to enable adaptive bitrate
        if isinstance(transport, DailyTransport):
            await transport.update_publishing(
                publishing_settings={
                    "camera": {
                        "sendSettings": {
                            "allowAdaptiveLayers": True,
                        }
                    }
                }
            )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
