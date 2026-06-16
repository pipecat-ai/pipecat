#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Grok Voice Agent Realtime Example

This example demonstrates using xAI's Grok Voice Agent API for real-time
voice conversations. The Grok Voice Agent provides:

- Real-time audio streaming with low latency
- Built-in voice activity detection (VAD)
- Multiple voice options (Ara, Rex, Sal, Eve, Leo)
- Built-in tools: web_search, x_search, file_search
- Custom function calling

Requirements:
    - XAI_API_KEY environment variable set
    - uv add "pipecat-ai[grok]"

Usage:
    python 50-grok-realtime.py --transport webrtc
    python 50-grok-realtime.py --transport daily
"""

import os
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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.xai.realtime.events import SessionProperties
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


# --- Function Handlers ---


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = 75 if format == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_current_time(params: FunctionCallParams):
    """Get the current time."""
    await params.result_callback(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timezone": "local",
        }
    )


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback(
        {
            "name": "The Golden Dragon",
            "cuisine": "Chinese",
            "location": location,
            "rating": 4.5,
        }
    )


# Create tools schema with custom functions


# --- Transport Configuration ---

# Note: We don't need local VAD since Grok has built-in server-side VAD.
# Audio sample rates are configured via PipelineParams, not transport params.
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
    logger.info("Starting Grok Voice Agent bot")

    # Configure Grok session properties
    session_properties = SessionProperties(
        # Voice options: Ara, Rex, Sal, Eve, Leo
        voice="Ara",
        # Grok-specific built-in tools can be added here:
        # tools=[
        #     WebSearchTool(),  # Enable web search
        #     XSearchTool(),    # Enable X/Twitter search
        # ],
    )

    # Create the Grok Realtime LLM service
    llm = GrokRealtimeLLMService(
        api_key=os.environ["XAI_API_KEY"],
        settings=GrokRealtimeLLMService.Settings(
            system_instruction="""You are a helpful and friendly AI assistant powered by Grok.

    You have access to several tools:
    - Weather information
    - Current time
    - Restaurant recommendations
    - Web search (built-in)
    - X/Twitter search (built-in)

    Your voice and personality should be warm and engaging. Keep your responses
    concise and conversational since this is a voice interaction.

    If the user asks about current events or news, use web search.
    If they ask about what people are saying on social media, use X search.

    Always be helpful and proactive in offering assistance.""",
            session_properties=session_properties,
        ),
    )

    # Register function handlers

    # Create context with initial message and tools
    context = LLMContext(
        [{"role": "developer", "content": "Say hello and introduce yourself!"}],
        [get_current_weather, get_current_time, get_restaurant_recommendation],
    )

    # It appears that Grok Realtime can sometimes be slow to detect the start
    # of a user's turn; uncomment the below imports and user_params to
    # enable "supplemental" interruptions.
    # from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.turns.user_turn_strategies import UserTurnStrategies
    # from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
    # from pipecat.turns.user_start.external_user_turn_start_strategy import (
    #     ExternalUserTurnStartStrategy,
    # )
    # from pipecat.turns.user_stop.external_user_turn_stop_strategy import (
    #     ExternalUserTurnStopStrategy,
    # )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
        # user_params=LLMUserAggregatorParams(
        #     vad_analyzer=SileroVADAnalyzer(),
        #     user_turn_strategies=UserTurnStrategies(
        #         start=[
        #             VADUserTurnStartStrategy(
        #                 enable_interruptions=True,
        #                 enable_user_speaking_frames=False,  # Grok already emits turn frames
        #             ),
        #             ExternalUserTurnStartStrategy(),
        #         ],
        #         stop=[ExternalUserTurnStopStrategy()],
        #     ),  # Grok already emits turn frames
        # ),
    )

    # Build the pipeline
    # Note: In realtime mode, transcription comes from Grok (upstream),
    # so transcript.user() goes BEFORE llm
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input (audio)
            user_aggregator,
            llm,  # Grok Realtime LLM (handles STT + LLM + TTS)
            transport.output(),  # Transport bot output (audio)
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
        # Kick off the conversation
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    # Subscribe to user turn lifecycle events. Grok emits its own
    # user-turn frames from server VAD, so on_user_turn_stopped fires at
    # the turn boundary. In realtime mode UserTurnStoppedMessage.content
    # is None because the user transcript isn't finalized at turn-stop
    # time — subscribe to on_user_turn_message_added for the finalized text
    # (it's written when the assistant response begins). The assistant
    # message is finalized at turn-stop time in both modes, so
    # on_assistant_turn_stopped carries the content directly.
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
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

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
