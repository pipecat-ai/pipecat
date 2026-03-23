#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example with IBM Watson Services.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. This version uses IBM Watson services for
speech-to-text and text-to-speech.

Required AI services:
- IBM Watson Speech-to-Text
- OpenAI (LLM)
- IBM Watson Text-to-Speech

Environment variables required:
- IBM_STT_API_KEY: Your IBM STT API key
- IBM_STT_URL: Your IBM STT service URL
- IBM_TTS_API_KEY: Your IBM TTS API key
- IBM_TTS_URL: Your IBM TTS service URL
- OPENAI_API_KEY: Your OpenAI API key

Run the bot using::

    uv run bot_ibm.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot with IBM Watson services...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

import sys
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pipecat.services.ibm.stt import WatsonSTTService
from pipecat.services.ibm.tts import WatsonTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot with IBM Watson services")

    # IBM Watson Speech-to-Text
    stt = WatsonSTTService(
        api_key=os.getenv("IBM_STT_API_KEY"),
        url=os.getenv("IBM_STT_URL"),
        # Optional: specify model, default is en-US_BroadbandModel
        model="en-US",
        params=WatsonSTTService.InputParams(
            interim_results=True,
            smart_formatting=True,
            timestamps=True,
            inactivity_timeout=-1,  # Disable inactivity timeout to prevent session timeouts
        ),
    )

    # IBM Watson Text-to-Speech
    tts = WatsonTTSService(
        api_key=os.getenv("IBM_TTS_API_KEY"),
        url=os.getenv("IBM_TTS_URL"),
        params=WatsonTTSService.InputParams(
            voice="en-US_EllieNatural",  # Default voice
            accept="audio/wav;rate=16000",  # Audio format
            # Optional: adjust speaking rate and pitch
            # rate_percentage=10,  # 10% faster
            # pitch_percentage=0,  # Default pitch
        ),
    )

    # OpenAI LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant powered by IBM Watson. Respond naturally and keep your answers conversational.",
        },
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # IBM Watson STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # IBM Watson TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": "Say hello and briefly introduce yourself as an AI assistant powered by IBM Watson.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

# Made with Bob
