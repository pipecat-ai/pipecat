#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    RealtimeServiceModeConfig,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


async def fetch_weather_from_api(params: FunctionCallParams):
    temperature = 75 if params.arguments["format"] == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": params.arguments["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def fetch_restaurant_recommendation(params: FunctionCallParams):
    await params.result_callback({"name": "The Golden Dragon"})


system_instruction = """
You are a helpful assistant who can answer questions and use tools.

You have three tools available to you:
1. get_current_weather: Use this tool to get the current weather in a specific location.
2. get_restaurant_recommendation: Use this tool to get a restaurant recommendation in a specific location.
3. google_search: Use this tool to search the web for information.
"""


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
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
    logger.info(f"Starting bot")

    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the user's location.",
            },
        },
        required=["location", "format"],
    )
    restaurant_function = FunctionSchema(
        name="get_restaurant_recommendation",
        description="Get a restaurant recommendation",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        required=["location"],
    )
    search_tool = {"google_search": {}}
    # KNOWN ISSUE: If using GeminiVertexLiveLLMService, it appears
    # you cannot use the "google_search" tool alongside other tools.
    # See https://github.com/googleapis/python-genai/issues/941.
    tools = ToolsSchema(
        standard_tools=[weather_function, restaurant_function],
        custom_tools={AdapterType.GEMINI: [search_tool]},
    )

    llm = GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GeminiLiveLLMService.Settings(
            system_instruction=system_instruction,
            voice="Aoede",  # Puck, Charon, Kore, Fenrir, Aoede
        ),
        tools=tools,
    )

    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("get_restaurant_recommendation", fetch_restaurant_recommendation)

    context = LLMContext()
    # Gemini Live drives the conversation server-side and does not emit
    # UserStartedSpeakingFrame / UserStoppedSpeakingFrame. Context
    # aggregation still works with realtime_service_mode, but pipeline
    # processors that depend on those frames (RTVI client speech events,
    # TurnTrackingObserver, AudioBufferProcessor turn recording,
    # UserIdleController, user mute strategies, voicemail detector) won't
    # activate. The Pipecat Prebuilt UI is one such consumer — without
    # these frames it can't group user transcripts into discrete turns
    # visually.
    #
    # If you need those frames, uncomment the SileroVADAnalyzer import
    # above and the `user_params=` argument below. Note: local turn
    # detection may not match Gemini Live's actual server-side turn
    # decisions and can desynchronize in subtle ways.
    #
    # For local VAD driving the conversation (server VAD disabled), see
    # `realtime-gemini-live-local-vad.py` instead.
    #
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.processors.aggregators.llm_response_universal import (
    #     LLMUserAggregatorParams,
    # )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=RealtimeServiceModeConfig(),
        # user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
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

    worker = PipelineWorker(
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
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    # Gemini Live doesn't emit user-turn frames so on_user_turn_stopped
    # would never fire. The *_message_added events fire when messages are
    # written to context and carry the finalized content; use those for
    # transcript logging regardless of whether the service emits turn
    # frames.
    @user_aggregator.event_handler("on_user_message_added")
    async def on_user_message_added(aggregator, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_message_added")
    async def on_assistant_message_added(aggregator, message: AssistantTurnStoppedMessage):
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
