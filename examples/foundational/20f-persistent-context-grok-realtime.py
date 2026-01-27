#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok Realtime persistent context example.

This example demonstrates how to save and load conversation history with
Grok's Realtime Voice Agent API. It allows:
- Saving the current conversation to a JSON file
- Loading a previous conversation from disk
- Listing all saved conversation files

This is useful for building voice agents that remember past conversations.
"""

import asyncio
import glob
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.grok.realtime.events import SessionProperties, TurnDetection
from pipecat.services.grok.realtime.llm import GrokRealtimeLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

BASE_FILENAME = "/tmp/pipecat_grok_conversation_"


async def fetch_weather_from_api(params: FunctionCallParams):
    """Mock weather function for demonstration."""
    temperature = 75 if params.arguments["format"] == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": params.arguments["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_saved_conversation_filenames(params: FunctionCallParams):
    """Get a list of saved conversation history files."""
    full_pattern = f"{BASE_FILENAME}*.json"
    matching_files = glob.glob(full_pattern)
    logger.debug(f"matching files: {matching_files}")
    await params.result_callback({"filenames": matching_files})


async def save_conversation(params: FunctionCallParams):
    """Save the current conversation to a JSON file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"{BASE_FILENAME}{timestamp}.json"
    logger.debug(
        f"writing conversation to {filename}\n{json.dumps(params.context.get_messages(), indent=4)}"
    )
    try:
        with open(filename, "w") as file:
            messages = params.context.get_messages()
            # Remove the last message (the save instruction)
            messages.pop()
            json.dump(messages, file, indent=2)
        await params.result_callback({"success": True})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def load_conversation(params: FunctionCallParams):
    """Load a conversation history from a JSON file."""

    async def _reset():
        filename = params.arguments["filename"]
        logger.debug(f"loading conversation from {filename}")
        try:
            with open(filename, "r") as file:
                params.context.set_messages(json.load(file))
                await params.llm.reset_conversation()
                # Manually create a response since we've reset the conversation
                await params.llm._create_response()
        except Exception as e:
            await params.result_callback({"success": False, "error": str(e)})

    asyncio.create_task(_reset())


# Define the tools schema
tools = ToolsSchema(
    standard_tools=[
        FunctionSchema(
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
        ),
        FunctionSchema(
            name="save_conversation",
            description="Save the current conversation. Use this function to persist the current conversation to external storage.",
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="get_saved_conversation_filenames",
            description="Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp.",
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="load_conversation",
            description="Load a conversation history. Use this function to load a conversation history into the current session.",
            properties={
                "filename": {
                    "type": "string",
                    "description": "The filename of the conversation history to load.",
                }
            },
            required=["filename"],
        ),
    ]
)


# Transport configuration - no local VAD needed since Grok has server-side VAD
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
    logger.info("Starting Grok Realtime persistent context bot")

    session_properties = SessionProperties(
        voice="Ara",
        turn_detection=TurnDetection(type="server_vad"),
        instructions="""You are a helpful and friendly AI assistant powered by Grok.

Your voice and personality should be warm and engaging, with a lively and playful tone.

You are participating in a voice conversation. Keep your responses concise, short, and to the point
unless specifically asked to elaborate on a topic.

You have access to tools for:
- Getting weather information
- Saving the current conversation to disk
- Loading previous conversations from disk
- Listing saved conversation files

When the user asks to save or load a conversation, use the appropriate tool.
Remember, your responses should be short - just one or two sentences usually.""",
    )

    llm = GrokRealtimeLLMService(
        api_key=os.getenv("GROK_API_KEY"),
        session_properties=session_properties,
    )

    # Register function handlers
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("save_conversation", save_conversation)
    llm.register_function("get_saved_conversation_filenames", get_saved_conversation_filenames)
    llm.register_function("load_conversation", load_conversation)

    context = LLMContext([{"role": "user", "content": "Say hello!"}], tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

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
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
