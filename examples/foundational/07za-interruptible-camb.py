#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Camb.ai MARS-8 TTS example with interruption handling.

This example demonstrates:
- Basic TTS synthesis with Camb.ai MARS-8
- Voice selection
- Speed control
- Handling interruptions

Requirements:
- CAMB_API_KEY environment variable
- OPENAI_API_KEY environment variable (for LLM)
- DEEPGRAM_API_KEY environment variable (for STT)

Usage:
    export CAMB_API_KEY=your_camb_api_key
    export OPENAI_API_KEY=your_openai_api_key
    export DEEPGRAM_API_KEY=your_deepgram_api_key
    python 07za-interruptible-camb.py --transport daily

For more information:
- Camb.ai API docs: https://camb.mintlify.app/
- Pipecat docs: https://docs.pipecat.ai/
"""

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.camb.tts import CambTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# Transport configuration for different platforms
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the bot with Camb.ai TTS.

    Args:
        transport: The transport to use for audio I/O.
        runner_args: Runner arguments from the CLI.
    """
    logger.info("Starting Camb.ai TTS bot")

    # Create an HTTP session for the TTS service
    async with aiohttp.ClientSession() as session:
        # Initialize Deepgram STT for speech recognition
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        # Initialize Camb.ai TTS with MARS-8-flash model (fastest)
        tts = CambTTSService(
            api_key=os.getenv("CAMB_API_KEY"),
            aiohttp_session=session,
            voice_id=2681,  # Attic voice (default)
            model="mars-8-flash",  # Fast inference model
            params=CambTTSService.InputParams(
                speed=1.0,  # Normal speed (0.5-2.0 range)
            ),
        )

        # Initialize OpenAI LLM
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        # System prompt for the assistant
        messages = [
            {
                "role": "system",
                "content": """You are a helpful voice assistant powered by Camb.ai's MARS-8
text-to-speech technology. Your goal is to have natural conversations and demonstrate
high-quality speech synthesis. Keep your responses concise and conversational since
they will be spoken aloud. Avoid special characters, emojis, or bullet points that
can't easily be spoken.""",
            },
        ]

        # Set up context management
        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)

        # Build the pipeline
        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # Speech-to-text
                context_aggregator.user(),  # User context aggregation
                llm,  # Language model
                tts,  # Camb.ai TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant context aggregation
            ]
        )

        # Create the pipeline task
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
            logger.info("Client connected")
            # Start the conversation with a greeting
            messages.append(
                {
                    "role": "system",
                    "content": "Please introduce yourself briefly and ask how you can help.",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        # Run the pipeline
        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud.

    Args:
        runner_args: Arguments passed from the runner.
    """
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


async def list_available_voices():
    """Helper function to list available Camb.ai voices.

    Run this to see what voices are available for your API key.
    """
    async with aiohttp.ClientSession() as session:
        voices = await CambTTSService.list_voices(
            api_key=os.getenv("CAMB_API_KEY"),
            aiohttp_session=session,
        )
        print("\nAvailable Camb.ai voices:")
        print("-" * 50)
        for voice in voices:
            print(f"  ID: {voice['id']}, Name: {voice['name']}, Gender: {voice['gender']}")
        print("-" * 50)
        print(f"Total: {len(voices)} voices\n")


if __name__ == "__main__":
    import sys

    # If --list-voices flag is passed, list voices and exit
    if "--list-voices" in sys.argv:
        import asyncio

        asyncio.run(list_available_voices())
    else:
        from pipecat.runner.run import main

        main()
