#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.cerebras import CerebrasLLMService
from pipecat.services.openai import OpenAILLMContext
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def start_fetch_weather(function_name, llm, context):
    # note: we can't push a frame to the LLM here. the bot
    # can interrupt itself and/or cause audio overlapping glitches.
    # possible question for Aleix and Chad about what the right way
    # to trigger speech is, now, with the new queues/async/sync refactors.
    # await llm.push_frame(TextFrame("Let me check on that."))
    logger.debug(f"Starting fetch_weather_from_api with function_name: {function_name}")


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    await result_callback({"conditions": "nice", "temperature": "75"})


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = CerebrasLLMService(api_key=os.getenv("CEREBRAS_API_KEY"), model="llama-3.3-70b")
        # Register a function_name of None to get all functions
        # sent to the same callback with an additional function_name parameter.
        llm.register_function(None, fetch_weather_from_api, start_callback=start_fetch_weather)

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_current_weather",
                    "description": "Get the current weather for a specific location. You MUST use this function whenever asked about weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Use fahrenheit for US locations, celsius for others.",
                            },
                        },
                        "required": ["location", "format"],
                    },
                },
            )
        ]
        messages = [
            {
                "role": "system",
                "content": """You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. 

You have one functions available:

1. get_current_weather is used to get current weather information.

Infer whether to use Fahrenheit or Celsius automatically based on the location, unless the user specifies a preference.

Start by asking me for my location. Then, use 'get_weather_current' to give me a forecast.

        Respond to what the user said in a creative and helpful way.""",
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
