#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


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
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Load environment variables
load_dotenv(override=True)


async def fetch_weather_from_api(params: FunctionCallParams):
    temperature = (
        random.randint(60, 85)
        if params.arguments["format"] == "fahrenheit"
        else random.randint(15, 30)
    )
    # Simulate a long network delay.
    # You can continue chatting while waiting for this to complete.
    # With Nova 2 Sonic (the default model), the assistant will respond
    # appropriately once the function call is complete.
    await asyncio.sleep(5)
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

# Create tools schema
tools = ToolsSchema(standard_tools=[weather_function])


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Specify initial system instruction.
    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken dialog exchanging "
        "the transcripts of a natural real-time conversation. Keep your responses short, generally "
        "two or three sentences for chatty scenarios."
        # HACK: if using the older Nova Sonic (pre-2) model, note that you need to inject a special
        # bit of text into this instruction to allow the first assistant response to be
        # programmatically triggered (which happens in the on_client_connected handler)
        # f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )

    # Create the AWS Nova Sonic LLM service
    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        # as of 2025-12-09, these are the supported regions:
        # - Nova 2 Sonic (the default model):
        #   - us-east-1
        #   - us-west-2
        #   - ap-northeast-1
        # - Nova Sonic (the older model):
        #   - us-east-1
        #   - ap-northeast-1
        region=os.getenv("AWS_REGION"),
        session_token=os.getenv("AWS_SESSION_TOKEN"),
        voice_id="tiffany",
        # you could choose to pass instruction here rather than via context
        # system_instruction=system_instruction
        # you could choose to pass tools here rather than via context
        # tools=tools
    )

    # Register function for function calls
    # you can either register a single function for all function calls, or specific functions
    # llm.register_function(None, fetch_weather_from_api)
    llm.register_function(
        "get_current_weather", fetch_weather_from_api, cancel_on_interruption=False
    )

    # Set up context and context management.
    context = LLMContext(
        messages=[
            {"role": "system", "content": f"{system_instruction}"},
            {
                "role": "user",
                "content": "Tell me a fun fact!",
            },
        ],
        tools=tools,
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # Configure the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Handle client connection event
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])
        # HACK: if using the older Nova Sonic (pre-2) model, you need this special way of
        # triggering the first assistant response. Note that this trigger requires a special
        # corresponding bit of text in the system instruction.
        # await llm.trigger_assistant_response()

    # Handle client disconnection events
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

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

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
