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
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import EndWorkerFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
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


async def end_conversation(params: FunctionCallParams):
    await params.result_callback({"success": True})
    await params.llm.push_frame(EndWorkerFrame())


# NOTE: we can ask the model to say something *after* the call to
# end_conversation because GeminiLiveLLMService defers processing EndFrames
# until after the bot finishes its current turn. With Gemini 3.1 Flash Live,
# the model won't reliably report ending its turn until after it says something
# following the tool call, which is why the system instruction is structured
# the way it is.
system_instruction = """
You are a helpful assistant who can answer questions and use tools.

You have three tools available to you:
1. get_current_weather: Use this tool to get the current weather in a specific location.
2. get_restaurant_recommendation: Use this tool to get a restaurant recommendation in a specific location.
3. end_conversation: Use this tool to gracefully end the conversation.

After you've responded to the user three times, do the following:
1. Politely let them know that that's all the time you have today (but don't say "goodbye" yet).
2. Then immediately call the end_conversation function. *DO NOT FORGET TO DO THIS STEP.*
3. After the tool reports success, say goodbye.
"""


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
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
        handler=fetch_weather_from_api,
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
        handler=fetch_restaurant_recommendation,
    )
    end_conversation_function = FunctionSchema(
        name="end_conversation",
        description="Gracefully end the conversation",
        properties={},
        required=[],
        handler=end_conversation,
    )
    search_tool = {"google_search": {}}
    tools = ToolsSchema(
        standard_tools=[weather_function, restaurant_function, end_conversation_function],
        custom_tools={AdapterType.GEMINI: [search_tool]},
    )

    llm = GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GeminiLiveLLMService.Settings(
            system_instruction=system_instruction,
        ),
        tools=tools,
    )

    context = LLMContext(
        [{"role": "developer", "content": "Say hello."}],
    )
    # Gemini Live doesn't emit user-turn frames. Server-side VAD is
    # enabled by default; to surface turn frames (for RTVI speech
    # events, turn observers, etc.) uncomment the local-VAD imports +
    # `user_params=` below. See realtime-gemini-live.py for the full
    # discussion.
    #
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.processors.aggregators.llm_response_universal import (
    #     LLMUserAggregatorParams,
    # )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
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
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

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
