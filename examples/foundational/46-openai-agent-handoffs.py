#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Advanced OpenAI Agent service example with handoffs.

This example demonstrates how to use multiple agents with handoffs in the
OpenAI Agents SDK within a Pipecat pipeline, showcasing agent orchestration
and specialization.

Requirements:
- OpenAI API key
- OpenAI Agents SDK: pip install openai-agents
"""

import os
import random
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from pipecat.frames.frames import LLMRunFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai_agent.agent_service import OpenAIAgentService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# Transport configuration
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True, audio_in_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True, audio_in_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True, audio_in_enabled=True),
}


def create_weather_tools():
    """Create weather-related tools."""

    def get_weather(location: str) -> str:
        """Get current weather for a location."""
        conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
        temp = random.randint(-10, 35)
        condition = random.choice(conditions)
        return f"The weather in {location} is {condition} with a temperature of {temp}°C."

    def get_forecast(location: str, days: int = 3) -> str:
        """Get weather forecast for multiple days."""
        forecast = []
        for i in range(days):
            conditions = ["sunny", "cloudy", "rainy", "snowy"]
            temp = random.randint(-5, 30)
            condition = random.choice(conditions)
            day = "today" if i == 0 else f"in {i} day{'s' if i > 1 else ''}"
            forecast.append(f"{day.capitalize()}: {condition}, {temp}°C")
        return f"Weather forecast for {location}:\n" + "\n".join(forecast)

    return [get_weather, get_forecast]


def create_trivia_tools():
    """Create trivia and fact tools."""

    def get_random_fact() -> str:
        """Get a random interesting fact."""
        facts = [
            "Honey never spoils. Archaeologists have found edible honey in ancient Egyptian tombs.",
            "A group of flamingos is called a 'flamboyance'.",
            "Octopuses have three hearts and blue blood.",
            "The Great Wall of China isn't visible from space with the naked eye.",
            "Bananas are berries, but strawberries aren't.",
            "Wombat poop is cube-shaped.",
            "A shrimp's heart is in its head.",
            "It's impossible to hum while holding your nose.",
        ]
        return random.choice(facts)

    def get_science_fact() -> str:
        """Get a random science fact."""
        facts = [
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "DNA stands for Deoxyribonucleic Acid.",
            "The human brain uses about 20% of the body's total energy.",
            "There are more possible games of chess than atoms in the observable universe.",
            "A single bolt of lightning contains enough energy to toast 100,000 slices of bread.",
        ]
        return random.choice(facts)

    return [get_random_fact, get_science_fact]


def create_math_tools():
    """Create math calculation tools."""

    def calculate(expression: str) -> str:
        """Safely calculate a mathematical expression."""
        try:
            # Only allow basic math operations for safety
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Sorry, I can only calculate basic math expressions with +, -, *, /, and parentheses."

            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

    def generate_math_problem() -> str:
        """Generate a random math problem."""
        operations = ["+", "-", "*"]
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(operations)

        if op == "+":
            answer = a + b
        elif op == "-":
            answer = a - b
        else:  # multiplication
            answer = a * b

        return f"Here's a math problem for you: {a} {op} {b} = ?"

    return [calculate, generate_math_problem]


async def create_specialist_agents():
    """Create specialized agents for different domains."""

    # Weather specialist agent
    weather_agent = OpenAIAgentService(
        name="Weather Specialist",
        instructions="""You are a weather specialist. You provide detailed weather information,
        forecasts, and weather-related advice. Use your tools to get accurate weather data.
        Be informative and helpful about weather conditions and what they might mean for
        outdoor activities.""",
        tools=create_weather_tools(),
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )

    # Trivia specialist agent
    trivia_agent = OpenAIAgentService(
        name="Trivia Master",
        instructions="""You are a trivia and facts specialist. You love sharing interesting
        facts, trivia, and educational content. Use your tools to provide fascinating
        information and engage users with fun facts. Make learning enjoyable!""",
        tools=create_trivia_tools(),
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )

    # Math specialist agent
    math_agent = OpenAIAgentService(
        name="Math Helper",
        instructions="""You are a mathematics specialist. You help with calculations,
        math problems, and mathematical concepts. Use your tools to solve problems
        and generate practice questions. Make math accessible and fun!""",
        tools=create_math_tools(),
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )

    return weather_agent, trivia_agent, math_agent


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting OpenAI Agent bot with handoffs")

    # Set up STT for speech recognition
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        model="nova-2",
    )

    # Set up TTS for voice output
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # Create specialist agents
    weather_agent, trivia_agent, math_agent = await create_specialist_agents()

    # Create the main triage agent that can hand off to specialists
    triage_agent = OpenAIAgentService(
        name="Assistant Coordinator",
        instructions="""You are a helpful assistant coordinator. Your role is to understand
        what the user needs and direct them to the right specialist:
        
        - For weather questions, forecasts, or outdoor activity planning -> Weather Specialist
        - For interesting facts, trivia, or educational content -> Trivia Master  
        - For calculations, math problems, or mathematical help -> Math Helper
        
        If the request doesn't clearly fit a specialist, you can handle general conversation
        yourself. Always be friendly and explain when you're connecting them to a specialist.""",
        handoffs=[weather_agent.agent, trivia_agent.agent, math_agent.agent],  # type: ignore
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )

    # Set up conversation context with initial system message
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful assistant coordinator with access to weather information, trivia, and math tools. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = triage_agent.create_context_aggregator(context)

    # Create the processing pipeline with context aggregators
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech to text
            context_aggregator.user(),  # User responses
            triage_agent,  # OpenAI Agent processing
            tts,  # Text to speech
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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
        # Kick off the conversation by adding system message and running LLM
        messages.append(
            {
                "role": "system",
                "content": "Please introduce yourself to the user as an AI assistant coordinator who works with specialists for weather, trivia, and math topics.",
            }
        )
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
