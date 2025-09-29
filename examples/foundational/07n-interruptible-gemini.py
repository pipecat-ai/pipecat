#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
A conversational AI bot using Gemini for both LLM and TTS.

This example demonstrates how to use Gemini's TTS capabilities with the new
GeminiTTSService, which uses Gemini's TTS-specific models instead of Google Cloud TTS.

Features showcased:
- Gemini LLM for conversation
- Gemini TTS with natural voice control
- Support for different voice personalities
- Style and tone control through natural language prompts

Run with:
    python examples/foundational/gemini-tts.py

Make sure to set your environment variables:
    export GOOGLE_API_KEY=your_api_key_here
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GeminiTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot with Gemini TTS")

    stt = GoogleSTTService(
        params=GoogleSTTService.InputParams(languages=Language.EN_US),
        credentials=os.getenv("GOOGLE_TEST_CREDENTIALS"),
    )

    tts = GeminiTTSService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-preview-tts",  # TTS-specific model
        voice_id="Charon",
        params=GeminiTTSService.InputParams(language=Language.EN_US),
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
    )

    # System message that instructs the AI on how to speak
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way.

            IMPORTANT: Since you're using Gemini TTS which supports natural voice control, you can include speaking instructions in your responses. For example:
            - "Say cheerfully: Welcome to our conversation!"
            - "Read this in a calm, professional tone: Here are the details you requested."
            - "Speak in an excited whisper: I have some great news to share!"
            - "Say slowly and clearly: Let me explain this step by step."

            Feel free to use natural language instructions to control your voice style, tone, pace, and emotion. The TTS system will interpret these instructions and adjust the speech accordingly.

            Your output will be converted to audio, so avoid special characters in your answers. Respond to what the user said in a creative and helpful way.""",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # Gemini TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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
        # Kick off the conversation with a styled introduction
        messages.append(
            {
                "role": "system",
                "content": "Say cheerfully and warmly: Hello! I'm your AI assistant powered by Gemini's new TTS technology. I can speak with different voices, tones, and styles. How can I help you today?",
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
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
