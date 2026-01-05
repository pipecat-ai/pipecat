#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Camb.ai MARS-8 TTS example with local audio (microphone/speakers).

This example demonstrates:
- Basic TTS synthesis with Camb.ai MARS-8
- Local audio input/output (no WebRTC or Daily needed)
- Handling interruptions

Requirements:
- CAMB_API_KEY environment variable
- OPENAI_API_KEY environment variable (for LLM)
- DEEPGRAM_API_KEY environment variable (for STT)

Usage:
    export CAMB_API_KEY=your_camb_api_key
    export OPENAI_API_KEY=your_openai_api_key
    export DEEPGRAM_API_KEY=your_deepgram_api_key
    python 07zb-interruptible-camb-local.py [--voice-id VOICE_ID]
"""

import argparse
import asyncio
import os
import sys

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
from pipecat.services.camb.tts import CambTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(voice_id: int):
    # Local audio transport - uses your microphone and speakers
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        )
    )

    # Deepgram STT for speech recognition
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Create HTTP session for Camb.ai TTS
    async with aiohttp.ClientSession() as session:
        # Camb.ai TTS with MARS-8-flash model
        tts = CambTTSService(
            api_key=os.getenv("CAMB_API_KEY"),
            aiohttp_session=session,
            voice_id=voice_id,
            model="mars-8-flash",
            params=CambTTSService.InputParams(
                speed=1.0,
            ),
        )

        # OpenAI LLM
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        # System prompt
        messages = [
            {
                "role": "system",
                "content": """You are a helpful voice assistant powered by Camb.ai's MARS-8
text-to-speech technology. Keep your responses concise and conversational since
they will be spoken aloud. Avoid special characters, emojis, or bullet points.""",
            },
        ]

        # Context management
        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)

        # Build the pipeline
        pipeline = Pipeline(
            [
                transport.input(),  # Microphone input
                stt,  # Speech-to-text
                context_aggregator.user(),  # User context
                llm,  # Language model
                tts,  # Camb.ai TTS
                transport.output(),  # Speaker output
                context_aggregator.assistant(),  # Assistant context
            ]
        )

        # Create pipeline task
        # Use 24kHz sample rate to match Camb.ai TTS output
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_out_sample_rate=24000,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Run the pipeline
        runner = PipelineRunner()
        logger.info("Starting Camb.ai TTS bot with local audio...")
        logger.info("Speak into your microphone to interact with the bot.")

        # Start the conversation with a greeting after a short delay
        async def start_greeting():
            await asyncio.sleep(1)  # Wait for pipeline to start
            messages.append(
                {
                    "role": "system",
                    "content": "Please introduce yourself briefly and ask how you can help.",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        # Run greeting and pipeline concurrently
        await asyncio.gather(
            runner.run(task),
            start_greeting(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camb.ai TTS example with local audio")
    parser.add_argument(
        "--voice-id",
        type=int,
        default=2681,
        help="Camb.ai voice ID to use (default: 2681 - Attic voice)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.voice_id))
