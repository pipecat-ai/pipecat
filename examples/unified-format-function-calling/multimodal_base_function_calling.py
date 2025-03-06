#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys
from typing import List

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.ai_services import LLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

load_dotenv(override=True)


async def start_fetch_weather(function_name, llm, context):
    """Push a frame to the LLM; this is handy when the LLM response might take a while."""
    await llm.push_frame(TTSSpeakFrame("Let me check on that."))
    logger.debug(f"Starting fetch_weather_from_api with function_name: {function_name}")


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    await result_callback({"conditions": "nice", "temperature": "75"})


class MultimodalWeatherBot:
    """Generic base class for setting up and running an LLM-powered bot."""

    def __init__(self, llm: LLMService):
        """Initialize the base handler with a specific LLM."""
        self.llm = llm

    @staticmethod
    def tools() -> ToolsSchema:
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
            required=["location"],
        )
        return ToolsSchema(standard_tools=[weather_function])

    async def run(self):
        """Set up and start the processing pipeline."""
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)

            transport = DailyTransport(
                room_url,
                token,
                "Respond bot",
                DailyParams(
                    audio_out_enabled=True,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                    vad_audio_passthrough=True,
                ),
            )

            # Register a function_name of None to get all functions
            # sent to the same callback with an additional function_name parameter.
            self.llm.register_function(
                None, fetch_weather_from_api, start_callback=start_fetch_weather
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who can report the weather in any location in the universe. Respond concisely. Your response will be turned into speech so use only simple words and punctuation.",
                },
                {"role": "user", "content": " Start the conversation by introducing yourself."},
            ]

            context = OpenAILLMContext(messages, MultimodalWeatherBot.tools())
            context_aggregator = self.llm.create_context_aggregator(context)

            pipeline = Pipeline(
                [
                    transport.input(),
                    context_aggregator.user(),
                    self.llm,
                    transport.output(),
                    context_aggregator.assistant(),
                ]
            )

            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                ),
            )

            @transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                await transport.capture_participant_transcription(participant["id"])
                await task.queue_frames([context_aggregator.user().get_context_frame()])

            runner = PipelineRunner()
            await runner.run(task)
