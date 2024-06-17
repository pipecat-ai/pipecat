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
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.logger import FrameLogger
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def start_fetch_weather(llm):
    await llm.push_frame(TextFrame("Let me think."))


async def fetch_weather_from_api(llm, args):
    location = args.get("location")
    format = args.get("format")

    url = (
        f"http://api.openweathermap.org/data/2.5/weather?"
        f"q={location}&appid={os.getenv('OPENWEATHERMAP_API_KEY')}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()

    temp_k = data["main"]["temp"]

    if format == "celsius":
        temp = temp_k - 273.15
    elif format == "fahrenheit":
        temp = (temp_k - 273.15) * 9 / 5 + 32
    else:
        raise ValueError(f"Unknown format: {format}")

    conditions = data["weather"][0]["description"]

    return {"conditions": conditions, "temperature": temp}


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
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

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        llm.register_function(
            "get_current_weather",
            fetch_weather_from_api,
            start_callback=start_fetch_weather,
        )

        fl_in = FrameLogger("Inner")
        fl_out = FrameLogger("Outer")

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city e.g. 'London, UK' or 'Paris, France'. State or country information is omitted",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the users location.",
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
                "content": """
                    You are a helpful LLM participating in a WebRTC call. 
                    Your primary goal is to demonstrate your capabilities concisely. 
                    Since your responses will be converted to audio, avoid using special characters. 
                    Respond to user queries in a creative and helpful manner. 
                    For weather-related questions, provide the temperature, current conditions, and recommended actions. 
                    If the user's query is not about the weather, politely prompt them to ask a weather-related question instead.
                """
            }

        ]

        context = OpenAILLMContext(messages, tools)
        tma_in = LLMUserContextAggregator(context)
        tma_out = LLMAssistantContextAggregator(context)
        pipeline = Pipeline(
            [
                fl_in,
                transport.input(),
                tma_in,
                llm,
                fl_out,
                tts,
                transport.output(),
                tma_out,
            ]
        )

        task = PipelineTask(pipeline)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            username = participant.get("info").get("userName")
            # Kick off the conversation.
            await tts.say(
                f"Hi {username}! Ask me about the weather anywhere in the world."
            )

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
