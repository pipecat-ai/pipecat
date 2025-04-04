#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.ultravox.stt import UltravoxSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

# NOTE: This example requires GPU resources to run efficiently.
# The Ultravox model is compute-intensive and performs best with GPU acceleration.
# This can be deployed on cloud GPU providers like Cerebrium.ai for optimal performance.

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Want to initialize the ultravox processor since it takes time to load the model and dont
# want to load it every time the pipeline is run
ultravox_processor = UltravoxSTTService(
    model_name="fixie-ai/ultravox-v0_5-llama-3_1-8b",
    hf_token=os.getenv("HF_TOKEN"),
)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
                vad_audio_passthrough=True,
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.environ.get("CARTESIA_API_KEY"),
            voice_id="97f4b8fb-f2fe-444b-bb9a-c109783a857a",
        )

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                ultravox_processor,
                tts,  # TTS
                transport.output(),  # Transport bot output
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
