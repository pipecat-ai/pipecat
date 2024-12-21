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
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramTTSService
from pipecat.services.google import GoogleLLMService, GoogleLLMContext
from pipecat.services.openai import OpenAILLMContext
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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


# Not necessary for function calling, but useful in observation
# and debugging
class functionCallPipecatObserver(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, FunctionCallInProgressFrame):
            await self.push_frame(frame)
        elif isinstance(frame, FunctionCallResultFrame):
            logger.info(f"Function {frame.function_name} returned: {frame.result}")
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)


async def get_noaa_simple_weather(zip_code: str, **kwargs):
    logger.debug(f"noaa get simple weather for {zip_code}")
    n = NOAA()
    observations = n.get_observations(postalcode=zip_code, country="US", num_of_stations=1)
    for observation in observations:
        description = observation["textDescription"]
        celcius_temp = observation["temperature"]["value"]

    fahrenheit_temp = (celcius_temp * 9 / 5) + 32
    return description, fahrenheit_temp


async def get_weather(function_name, tool_call_id, arguments, llm, context, result_callback):
    location = arguments["location"]
    logger.debug(f"get_weather location: {location}")

    if len(location) == 5 and location.isdigit():
        description, fahrenheit_temp = await get_noaa_simple_weather(location)
    else:
        return await result_callback(
            f"I'm sorry, I can't get the weather for {location} right now. Can you ask again please?"
        )

    await result_callback(
        f"The weather in {location} is currently {round(fahrenheit_temp)} degrees and {description}."
    )


async def get_postalcode(function_name, tool_call_id, arguments, llm, context, result_callback):
    location = arguments["location"]
    await result_callback(f"{location}")


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

        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")

        llm = GoogleLLMService(
            model="gemini-1.5-flash-latest",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
        llm.register_function("get_weather", get_weather)
        llm.register_function("get_postalcode", get_postalcode)

        tools = [
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location for the weather request. Must be a ZIP code.",
                                },
                            },
                            "required": ["location"],
                        },
                    },
                    {
                        "name": "get_postalcode",
                        "description": "Get the postal code of a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location to provide a postalcode for.",
                                },
                                "zip": {
                                    "type": "string",
                                    "description": "Infer the postalcode from the location. Your options are any number between 00602 and 99999. Only respond with the 5 digit postal code.",
                                },
                            },
                            "required": ["location"],
                        },
                    },
                ]
            }
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Say hello and offer to provide weather information for anywhere in the United States of America.",
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        functionCallObserver = functionCallPipecatObserver()

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                functionCallObserver,
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
