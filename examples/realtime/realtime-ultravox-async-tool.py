#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: async function call with the Ultravox Realtime LLM service.

The ``get_current_weather`` tool is registered with
``cancel_on_interruption=False`` and simulates a slow API call (10s sleep).

Ultravox's API freezes the conversation between ``client_tool_invocation``
and the matching ``client_tool_result``, so the service ships a placeholder
``client_tool_result`` immediately when an async-registered function is
invoked (to unfreeze the conversation). When the real tool finishes, the
actual result is injected as user-side text so the model picks it up.
"""

import asyncio
import datetime
import os
import random

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.evals.transport import EvalTransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ultravox.llm import OneShotInputParams, UltravoxRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


@tool_options(cancel_on_interruption=False)
async def fetch_weather_from_api(params: FunctionCallParams):
    # Simulate a long-running API call so we can demonstrate that the
    # conversation continues while the tool is in flight.
    await asyncio.sleep(10)
    temperature = (
        random.randint(60, 85)
        if params.arguments["format"] == "fahrenheit"
        else random.randint(15, 30)
    )
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "location": params.arguments["location"],
            "format": params.arguments["format"],
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


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
            "description": "The temperature unit to use. Infer this from the users location.",
        },
    },
    required=["location", "format"],
    handler=fetch_weather_from_api,
)


system_prompt = (
    "You are a friendly assistant. The user and you will engage in a spoken "
    "dialog exchanging the transcripts of a natural real-time conversation. "
    "Keep your responses short, generally two or three sentences for chatty "
    "scenarios. When the user asks for the weather, call get_current_weather. "
    "While you wait for the result, keep chatting with the user. When the "
    "result arrives, share it with the user naturally."
)


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

    llm = UltravoxRealtimeLLMService(
        params=OneShotInputParams(
            api_key=os.environ["ULTRAVOX_API_KEY"],
            system_prompt=system_prompt,
            temperature=0.3,
            max_duration=datetime.timedelta(minutes=3),
        ),
        one_shot_selected_tools=[weather_function],
    )

    context = LLMContext([])
    # Ultravox doesn't emit user-turn frames. To get them (for RTVI
    # speech events, turn observers, etc.) uncomment the local-VAD
    # imports + `user_params=` below. See realtime-ultravox.py for the
    # full discussion.
    #
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.processors.aggregators.llm_response_universal import (
    #     LLMUserAggregatorParams,
    # )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
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
