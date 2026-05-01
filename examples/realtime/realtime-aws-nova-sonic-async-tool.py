#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: async function call with the AWS Nova Sonic LLM service.

The ``get_current_weather`` tool is registered with
``cancel_on_interruption=False`` and simulates a slow API call (20s sleep).
While the call is in flight the conversation continues; the result arrives
later via the async-tool mechanism and is forwarded to Nova Sonic so the
model can integrate it naturally into its next turn.
"""

import asyncio
import os
import random
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


async def fetch_weather_from_api(params: FunctionCallParams):
    # Simulate a long-running API call so we can demonstrate that the
    # conversation continues while the tool is in flight.
    await asyncio.sleep(20)
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
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
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
)

tools = ToolsSchema(standard_tools=[weather_function])


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

    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken "
        "dialog exchanging the transcripts of a natural real-time conversation. "
        "Keep your responses short, generally two or three sentences for chatty "
        "scenarios. When the user asks for the weather, call get_current_weather. "
        "While you wait for the result, keep chatting with the user. When the "
        "result arrives, share it with the user naturally."
    )

    llm = AWSNovaSonicLLMService(
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        region=os.environ["AWS_REGION"],
        session_token=os.getenv("AWS_SESSION_TOKEN"),
        settings=AWSNovaSonicLLMService.Settings(
            voice="tiffany",
            system_instruction=system_instruction,
        ),
    )

    llm.register_function(
        "get_current_weather",
        fetch_weather_from_api,
        cancel_on_interruption=False,
    )

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
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
