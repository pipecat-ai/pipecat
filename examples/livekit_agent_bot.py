#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.livekit_worker import LiveKitWorkerRunner
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.turns.user_turn_strategies import UserTurnStrategies

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def pipeline_factory(room):
    """
    Creates the pipeline for the agent using the connected LiveKit room.
    This function is called by LiveKitWorkerRunner when a job is assigned.
    """
    logger.info(f"Creating pipeline for room: {room.name}")

    # Inject the existing room into the transport
    transport = LiveKitTransport(
        url=room.url or "",  # Not strictly needed when room is injected, but good practice
        token="",  # Token managed by the Worker
        room_name=room.name,
        room=room,  # <--- Dependency Injection
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
    )

    messages = [
        {
            "role": "system",
            "content": "You are a specific assistant running as a LiveKit Agent Worker. Say hello!",
        },
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(log_turns=True)
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    return transport, pipeline


def main():
    # Initialize the Worker Runner
    runner = LiveKitWorkerRunner()

    # Register the factory that builds the pipeline per room
    runner.set_pipeline_factory(pipeline_factory)

    # Start the worker (blocks forever listening for jobs)
    runner.run()


if __name__ == "__main__":
    main()
