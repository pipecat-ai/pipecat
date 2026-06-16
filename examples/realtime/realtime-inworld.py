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
    - uv add "pipecat-ai[inworld]"

Usage:
    python realtime-inworld.py --transport webrtc
    python realtime-inworld.py --transport daily
"""

import os
import random
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import (
    TranscriptionLogObserver,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnMessageAddedMessage,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = random.randint(60, 85) if format == "fahrenheit" else random.randint(15, 30)
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "location": location,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


# --- Transport Configuration ---

# No local VAD needed — Inworld's server-side semantic VAD handles turn detection.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
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
        api_key=os.environ["INWORLD_API_KEY"],
        llm_model="google-ai-studio/gemini-3.1-flash-lite",
        voice="Sarah",
        settings=InworldRealtimeLLMService.Settings(
            system_instruction="""You are a helpful and friendly AI assistant powered by Inworld.

Your voice and personality should be warm and engaging. Keep your responses
concise and conversational since this is a voice interaction.

Always be helpful and proactive in offering assistance.""",
        ),
    )

    # Note: function calling requires a paid Inworld account and a
    # function-calling-capable model

    # Create context with initial message + tools
    context = LLMContext(
        [{"role": "developer", "content": "Say hello and introduce yourself!"}],
        [get_current_weather],
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
    )

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

    worker = PipelineWorker(
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
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    # Subscribe to user turn lifecycle events. Inworld emits its own
    # user-turn frames from server-side semantic VAD, so
    # on_user_turn_stopped fires at the turn boundary. In realtime mode
    # UserTurnStoppedMessage.content is None because the user transcript
    # isn't finalized at turn-stop time — subscribe to
    # on_user_turn_message_added for the finalized text (it's written when
    # the assistant response begins). The assistant message is finalized
    # at turn-stop time in both modes, so on_assistant_turn_stopped
    # carries the content directly.
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator,
        strategy: BaseUserTurnStopStrategy,
        message: UserTurnStoppedMessage,
    ):
        logger.info(f"User turn stopped at {message.timestamp}")

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}user: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}assistant: {message.content}")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
