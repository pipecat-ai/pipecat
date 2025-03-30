#
# Copyright (c) 2024–2025, Daily
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
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    # Add a delay to test interruption during function calls
    logger.info("Weather API call starting...")
    await asyncio.sleep(5)  # 5-second delay
    logger.info("Weather API call completed")
    await result_callback({"conditions": "nice", "temperature": "75"})


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        # Configure the mute processor with both strategies
        stt_mute_processor = STTMuteFilter(
            config=STTMuteConfig(
                strategies={
                    STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE,
                    STTMuteStrategy.FUNCTION_CALL,
                }
            ),
        )

        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        llm.register_function("get_current_weather", fetch_weather_from_api)

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
                                "description": "The city and state, e.g. San Francisco, CA",
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
                "content": "You are a helpful assistant who can check the weather. Always check the weather when a location is mentioned. Respond concisely and naturally. Your output will be converted to audio so use only simple words and punctuation.",
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt_mute_processor,  # Add the mute processor before STT
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Kick off the conversation with a weather-related prompt
            messages.append(
                {
                    "role": "system",
                    "content": "Ask the user what city they'd like to know the weather for.",
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
