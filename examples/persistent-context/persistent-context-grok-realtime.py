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

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.xai.realtime.events import SessionProperties, TurnDetection
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

BASE_FILENAME = "/tmp/pipecat_grok_conversation_"


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


async def get_saved_conversation_filenames(params: FunctionCallParams):
    """Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp. Each file is conversation history that can be loaded into this session."""
    full_pattern = f"{BASE_FILENAME}*.json"
    matching_files = glob.glob(full_pattern)
    logger.debug(f"matching files: {matching_files}")
    await params.result_callback({"filenames": matching_files})


async def save_conversation(params: FunctionCallParams):
    """Save the current conversation. Use this function to persist the current conversation to external storage."""
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


async def load_conversation(params: FunctionCallParams, filename: str):
    """Load a conversation history. Use this function to load a conversation history into the current session.

    Args:
        filename: The filename of the conversation history to load.
    """

    async def _reset():
        logger.debug(f"loading conversation from {filename}")
        try:
            with open(filename) as file:
                params.context.set_messages(json.load(file))
                await params.llm.reset_conversation()
                # Manually create a response since we've reset the conversation
                await params.llm._create_response()
        except Exception as e:
            await params.result_callback({"success": False, "error": str(e)})

    asyncio.create_task(_reset())


# Define the tools schema


# Transport configuration - no local VAD needed since Grok has server-side VAD
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
        api_key=os.environ["XAI_API_KEY"],
        session_properties=session_properties,
    )

    # Register function handlers

    context = LLMContext(
        [{"role": "developer", "content": "Say hello!"}],
        [
            get_current_weather,
            save_conversation,
            get_saved_conversation_filenames,
            load_conversation,
        ],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
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
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
