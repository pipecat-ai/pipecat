#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok Realtime with locally-driven turn detection.

By default Grok Realtime drives the conversation with its own server-side
VAD (see `realtime-grok.py`). This variant disables server-side turn
detection (``turn_detection=None``, the "manual" mode in Grok's session
properties) and instead drives turn boundaries locally with
``SileroVADAnalyzer`` wired into the user aggregator. Use this variant if
you want a turn analyzer like ``LocalSmartTurnV3`` to decide when the user
is done speaking, or if you need ``UserStartedSpeakingFrame`` /
``UserStoppedSpeakingFrame`` to fire from the same source as
``InterruptionFrame``.

Caveat: locally-generated turn boundaries are a heuristic and may not match
the provider's actual server-side turn decisions. Prefer server-emitted
turn frames (i.e. the base `realtime-grok.py` example) unless you have a
specific reason to drive turn detection locally.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
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
    LLMUserAggregatorParams,
    RealtimeServiceModeConfig,
    UserMessageAddedMessage,
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

load_dotenv(override=True)


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

tools = ToolsSchema(standard_tools=[weather_function, time_function, restaurant_function])


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

    session_properties = SessionProperties(
        voice="Ara",
        # Disable Grok's server-side turn detection (manual mode). This
        # example drives turn boundaries locally via the SileroVADAnalyzer
        # wired into the user aggregator below.
        turn_detection=None,
    )

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

    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("get_current_time", get_current_time)
    llm.register_function("get_restaurant_recommendation", get_restaurant_recommendation)

    context = LLMContext(
        [{"role": "developer", "content": "Say hello and introduce yourself!"}],
        tools,
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        # Drive turn detection locally via SileroVAD wired into the user
        # aggregator. realtime_service_mode keeps context-write semantics
        # correct and (by default) drops the transcript wait on turn-end so
        # local VAD can drive turn boundaries on the latency critical path.
        realtime_service_mode=RealtimeServiceModeConfig(),
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
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

    # In realtime mode the user transcript isn't finalized at turn-stop
    # time, so on_user_turn_stopped carries no content; subscribe to
    # on_user_message_added below for the finalized text.
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator,
        strategy: BaseUserTurnStopStrategy,
        message: UserTurnStoppedMessage,
    ):
        logger.info(f"User turn stopped at {message.timestamp}")

    # In realtime mode this is the canonical "user said X" event,
    # decoupled from turn-stop.
    @user_aggregator.event_handler("on_user_message_added")
    async def on_user_message_added(aggregator, message: UserMessageAddedMessage):
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
