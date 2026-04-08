#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Inworld Realtime Example

This example demonstrates using Inworld's Realtime API for real-time voice
conversations. The Inworld Realtime API is OpenAI-compatible and operates
as a cascade STT/LLM/TTS pipeline under the hood, with built-in semantic
voice activity detection for turn management.

Features:
- Real-time audio streaming with low latency
- Built-in semantic VAD (voice activity detection)
- Streaming user transcription
- Text and audio input

Requirements:
    - INWORLD_API_KEY environment variable set
    - pip install pipecat-ai[inworld]

Usage:
    python realtime-inworld.py --transport webrtc
    python realtime-inworld.py --transport daily
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import (
    TranscriptionLogObserver,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# --- Transport Configuration ---

# No local VAD needed — Inworld's server-side semantic VAD handles turn detection.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Inworld Realtime bot")

    # Create the Inworld Realtime LLM service.
    # Common params (llm_model, voice, tts_model, stt_model) are top-level.
    # For full control, use settings=InworldRealtimeLLMService.Settings(session_properties=...)
    #
    # llm_model can be any supported model or an Inworld Router.
    # See: https://docs.inworld.ai/router/introduction
    llm = InworldRealtimeLLMService(
        api_key=os.getenv("INWORLD_API_KEY"),
        llm_model="xai/grok-4-1-fast-non-reasoning",
        voice="Sarah",
        settings=InworldRealtimeLLMService.Settings(
            system_instruction="""You are a helpful and friendly AI assistant powered by Inworld.

Your voice and personality should be warm and engaging. Keep your responses
concise and conversational since this is a voice interaction.

Always be helpful and proactive in offering assistance.""",
        ),
    )

    # Create context with initial message
    context = LLMContext(
        [{"role": "developer", "content": "Say hello and introduce yourself!"}],
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,  # Inworld Realtime (handles STT + LLM + TTS)
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[TranscriptionLogObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}user: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}assistant: {message.content}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
