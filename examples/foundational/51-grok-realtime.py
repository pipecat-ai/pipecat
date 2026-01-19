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
    - pip install pipecat-ai[grok]

Usage:
    python 50-grok-realtime.py --transport webrtc
    python 50-grok-realtime.py --transport daily
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# Note: Grok has built-in server-side VAD, so we don't need local VAD
# from pipecat.audio.vad.silero import SileroVADAnalyzer
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
from pipecat.services.grok.realtime.events import (
    SessionProperties,
    WebSearchTool,
    XSearchTool,
)
from pipecat.services.grok.realtime.llm import GrokRealtimeLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# --- Function Handlers ---


async def fetch_weather_from_api(params: FunctionCallParams):
    """Handle weather function calls."""
    temperature = 75 if params.arguments.get("format") == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": params.arguments.get("format", "celsius"),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_current_time(params: FunctionCallParams):
    """Handle time function calls."""
    await params.result_callback(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timezone": "local",
        }
    )


async def get_restaurant_recommendation(params: FunctionCallParams):
    """Handle restaurant recommendation function calls."""
    location = params.arguments.get("location", "unknown")
    await params.result_callback(
        {
            "name": "The Golden Dragon",
            "cuisine": "Chinese",
            "location": location,
            "rating": 4.5,
        }
    )


# --- Function Schemas ---

weather_function = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather for a location",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use.",
        },
    },
    required=["location", "format"],
)

time_function = FunctionSchema(
    name="get_current_time",
    description="Get the current time and date",
    properties={},
    required=[],
)

restaurant_function = FunctionSchema(
    name="get_restaurant_recommendation",
    description="Get a restaurant recommendation for a location",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
    },
    required=["location"],
)

# Create tools schema with custom functions
tools = ToolsSchema(standard_tools=[weather_function, time_function, restaurant_function])


# --- Transport Configuration ---

# Note: We don't need local VAD since Grok has built-in server-side VAD.
# Audio sample rates are configured via PipelineParams, not transport params.
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
    logger.info("Starting Grok Voice Agent bot")

    # Configure Grok session properties
    session_properties = SessionProperties(
        # Voice options: Ara, Rex, Sal, Eve, Leo
        voice="Ara",
        # System instructions
        instructions="""You are a helpful and friendly AI assistant powered by Grok.

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
        # Grok-specific built-in tools can be added here:
        # tools=[
        #     WebSearchTool(),  # Enable web search
        #     XSearchTool(),    # Enable X/Twitter search
        # ],
    )

    # Create the Grok Realtime LLM service
    llm = GrokRealtimeLLMService(
        api_key=os.getenv("GROK_API_KEY"),
        session_properties=session_properties,
    )

    # Register function handlers
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("get_current_time", get_current_time)
    llm.register_function("get_restaurant_recommendation", get_restaurant_recommendation)

    # Create context with initial message and tools
    context = LLMContext(
        [{"role": "user", "content": "Say hello and introduce yourself!"}],
        tools,
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

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
        # Kick off the conversation
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # Log transcript updates
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
