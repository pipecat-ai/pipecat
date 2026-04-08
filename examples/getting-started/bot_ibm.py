#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example with IBM Speech Services.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. This version uses IBM Speech Services for
speech-to-text and text-to-speech.

Required AI services:
- IBM Speech-to-Text
- Google Gemini (LLM)
- IBM Text-to-Speech

Environment variables required:
- IBM_STT_API_KEY: Your IBM STT API key
- IBM_STT_URL: Your IBM STT service URL
- IBM_TTS_API_KEY: Your IBM TTS API key
- IBM_TTS_URL: Your IBM TTS service URL
- GOOGLE_API_KEY: Your Google API key

Run the bot using::

    uv run bot_ibm.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot with IBM Speech Services...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver

logger.info("Loading pipeline components...")
import sys

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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.ibm.stt import IBMSTTService
from pipecat.services.ibm.tts import IBMTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot with IBM Speech Services")

    # IBM Speech-to-Text
    stt = IBMSTTService(
        api_key=os.getenv("IBM_STT_API_KEY"),
        url=os.getenv("IBM_STT_URL"),
        # Optional: specify model, default is en-US_BroadbandModel
        model="en-US",
        params=IBMSTTService.InputParams(
            interim_results=True,
            smart_formatting=True,
            timestamps=True,
            inactivity_timeout=-1,  # Disable inactivity timeout to prevent session timeouts
        ),
    )

    # IBM Text-to-Speech
    tts = IBMTTSService(
        api_key=os.getenv("IBM_TTS_API_KEY"),
        url=os.getenv("IBM_TTS_URL"),
        params=IBMTTSService.InputParams(
            voice="en-US_EmmaNatural",  # Default voice
            accept="audio/l16;rate=24000",  # Audio format
            # Optional: adjust speaking rate and pitch
            # rate_percentage=10,  # 10% faster
            # pitch_percentage=0,  # Default pitch
        ),
    )

    # Google Gemini LLM
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-lite"  # Higher quota: 1500 requests/day vs 20 for gemini-2.5-flash-lite
    )

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant powered by IBM Speech Services. Respond naturally and keep your answers conversational.",
        },
    ]

    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
     #   user_kwargs={"aggregation_timeout": 0},  # Remove 1-second latency
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # IBM STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # IBM TTS
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

    # Add UserBotLatencyObserver to track user-to-bot latency with per-service breakdown
    latency_observer = UserBotLatencyObserver()
    
    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds: float):
        """Called when overall user-to-bot latency is measured."""
        logger.info(f"🎯 User-to-bot latency: {latency_seconds:.3f}s ({latency_seconds * 1000:.0f}ms)")
    
    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        """Called with per-service latency breakdown."""
        logger.info("📊 Latency breakdown:")
        for event in breakdown.chronological_events():
            logger.info(f"  {event}")
    
    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(observer, latency_seconds: float):
        """Called when time to first bot speech is measured."""
        logger.info(f"🗣️  First bot speech: {latency_seconds:.3f}s ({latency_seconds * 1000:.0f}ms) after client connected")
    
    # Add observer to task
    task.add_observer(latency_observer)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": "Say hello and briefly introduce yourself as an AI assistant powered by IBM Speech Services.",
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

