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
from pipecat.services.nim import NimLLMService
from pipecat.services.openai import OpenAILLMContext
from pipecat.transports.services.daily import DailyParams, DailyTransport

try:
    from noaa_sdk import NOAA

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to run this example, please run `pip install noaa_sdk` and try again.")
    raise Exception(f"Missing module: {e}")

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

system_prompt = """\
You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions.

Your response will be turned into speech so use only simple words and punctuation.

You have access to two tools: get_weather and get_postalcode.

You can respond to questions about the weather using the get_weather tool.
When you are asked about the weather, infer from the location what the postal code is and use that as the zip_code argument in the get_weather tool.
"""


async def start_fetch_weather(function_name, llm, context):
    # note: we can't push a frame to the LLM here. the bot
    # can interrupt itself and/or cause audio overlapping glitches.
    # possible question for Aleix and Chad about what the right way
    # to trigger speech is, now, with the new queues/async/sync refactors.
    # await llm.push_frame(TextFrame("Let me check on that."))
    logger.debug(f"Starting fetch_weather_from_api with function_name: {function_name}")


async def get_noaa_simple_weather(zip_code: str, **kwargs):
    logger.debug(f"noaa get simple weather for {zip_code}")
    n = NOAA()
    observations = n.get_observations(postalcode=zip_code, country="US", num_of_stations=1)
    for observation in observations:
        description = observation["textDescription"]
        celcius_temp = observation["temperature"]["value"]

    fahrenheit_temp = (celcius_temp * 9 / 5) + 32
    return description, fahrenheit_temp


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    location = args["location"]
    zip_code = args["zip_code"]
    logger.info(f"fetch_weather_from_api * location: {location}, zip_code: {zip_code}")

    if len(zip_code) == 5 and zip_code.isdigit():
        description, fahrenheit_temp = await get_noaa_simple_weather(zip_code)
    else:
        return await result_callback(
            f"I'm sorry, I can't get the weather for {location} right now. Can you ask again please?"
        )

    await result_callback(
        f"The weather in {location} is currently {round(fahrenheit_temp)} degrees and {description}."
    )


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "weather bot",
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

        llm = NimLLMService(
            api_key=os.getenv("NVIDIA_API_KEY"), model="meta/llama-3.3-70b-instruct"
        )
        # Register a function_name of None to get all functions
        # sent to the same callback with an additional function_name parameter.
        llm.register_function(None, fetch_weather_from_api, start_callback=start_fetch_weather)

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location for the weather request.",
                            },
                            "zip_code": {
                                "type": "string",
                                "description": "The location for the weather request. Must only be a 5 digit postal code.",
                            },
                        },
                        "required": ["location"],
                    },
                },
            ),
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_postalcode",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to provide a postalcode for.",
                            },
                            "zip_code": {
                                "type": "string",
                                "description": "Infer the postalcode from the location. Your options are any number between 00602 and 99999. Only respond with the 5 digit postal code.",
                            },
                        },
                        "required": ["location"],
                    },
                },
            ),
        ]
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "Say hello and offer to provide weather information for anywhere in the United States of America.",
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
