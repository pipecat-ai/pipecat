#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from strands import Agent, tool
from strands.models import BedrockModel

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)

"""This example demonstrates how to use the Strands agent with Pipecat.

You can delegate complex, multi-step tasks to the Strands agent, which can cycle through LLM-based reasoning and tool calls to accomplish the task.

Try asking: "What's the weather where the Golden Gate Bridge is?"
"""

# Strands agent tools


@tool
def get_location_name_from_landmark(landmark: str) -> str:
    """
    Get the location name from a landmark.

    Args:
        landmark (str): The name of the landmark, e.g. "Golden Gate Bridge".
    """
    # Simulate fetching location
    return "San Francisco, CA"


@tool
def get_lat_long_from_location_name(location: str) -> dict:
    """
    Get the latitude and longitude for a location name.

    Args:
        location (str): The city and state, e.g. "San Francisco, CA".
    """
    # Simulate fetching lat/long from a geocoding service
    return {"lat": 37.7749, "long": -122.4194}


@tool
def get_current_weather_from_lat_long(lat: float, long: float) -> dict:
    """
    Get the current weather for a specific latitude and longitude.

    Args:
        lat (float): The latitude of the location.
        long (float): The longitude of the location.
    """
    # Simulate fetching weather data from a weather service
    return {"conditions": "nice", "temperature": "75"}


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
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


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    strands_agent = Agent(
        model=BedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", max_tokens=64000
        ),
        tools=[
            get_location_name_from_landmark,
            get_lat_long_from_location_name,
            get_current_weather_from_lat_long,
        ],
        system_prompt="""
        You are a helpful personal assistant who can look up information about places and weather.

        Your key capabilities:
        1. Look up where landmarks are located.
        2. Find latitude and longitude for a location.
        3. Look up the current weather for a specific latitude and longitude.

        Explain each step of your reasoning in clear, simple, and concise language. Your responses will be converted to audio, so avoid special characters and numbered lists.
        """,
    )

    async def handle_location_or_weather_related_queries(params: FunctionCallParams, query: str):
        """
        Handle location or weather related queries.

        Args:
            query (str): The user's query, e.g. "What's the weather where the Golden Gate Bridge is?".
        """
        # Run in a background thread
        # (Otherwise the agent blocks the event loop; one effect of that is that we don't hear
        # "let me check on that" until the agent finishes)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, strands_agent, query)
        await params.result_callback(result.message)

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    llm.register_direct_function(handle_location_or_weather_related_queries)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    tools = ToolsSchema(standard_tools=[handle_location_or_weather_related_queries])

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way. Start by suggesting that the user ask about the weather where the Golden Gate Bridge is.",
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
