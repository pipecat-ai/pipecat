#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Basic OpenAI Agent service example.

This example demonstrates how to use the OpenAI Agents SDK within a Pipecat
pipeline to create an interactive agent with tool calling capabilities.

Requirements:
- OpenAI API key
- OpenAI Agents SDK: pip install openai-agents
"""

import os
import random

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai_agent.agent_service import OpenAIAgentService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# Transport configuration
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


def get_weather_tool():
    """Example tool function for weather information."""

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city or location to get weather for.

        Returns:
            A weather description string.
        """
        # Simulate weather data
        conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
        temp = random.randint(-10, 35)
        condition = random.choice(conditions)

        return f"The weather in {location} is {condition} with a temperature of {temp}°C."

    return get_weather


def get_random_fact_tool():
    """Example tool function for random facts."""

    def get_random_fact() -> str:
        """Get a random interesting fact.

        Returns:
            A random fact string.
        """
        facts = [
            "Honey never spoils. Archaeologists have found edible honey in ancient Egyptian tombs.",
            "A group of flamingos is called a 'flamboyance'.",
            "Octopuses have three hearts and blue blood.",
            "The Great Wall of China isn't visible from space with the naked eye.",
            "Bananas are berries, but strawberries aren't.",
        ]
        return random.choice(facts)

    return get_random_fact


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting OpenAI Agent bot")

    # Set up TTS for voice output
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # Create tools for the agent
    tools = [
        get_weather_tool(),
        get_random_fact_tool(),
    ]

    # Initialize the OpenAI Agent service
    agent_service = OpenAIAgentService(
        name="Assistant",
        instructions="""You are a helpful assistant with access to weather information and random facts. 
        You can:
        - Check weather for any location using the get_weather tool
        - Share interesting facts using the get_random_fact tool
        - Have natural conversations
        
        Be friendly, informative, and engaging in your responses.""",
        tools=tools,
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )

    # Create the processing pipeline
    pipeline = Pipeline(
        [
            agent_service,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Send an initial greeting when client connects
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected, sending greeting")
        await task.queue_frames(
            [
                TextFrame(
                    "Hello! I'm an AI assistant powered by the OpenAI Agents SDK. "
                    "I can help you with weather information, share interesting facts, "
                    "or just have a conversation. What would you like to know?"
                ),
                EndFrame(),
            ]
        )

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
