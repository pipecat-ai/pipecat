#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from openai.types.chat import ChatCompletionToolParam

from runner import configure

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

current_voice = "News Lady"


async def switch_voice(function_name, tool_call_id, args, llm, context, result_callback):
    global current_voice
    current_voice = args["voice"]
    await result_callback(
        {
            "voice": f"You are now using your {current_voice} voice. Your responses should now be as if you were a {current_voice}."
        }
    )


async def news_lady_filter(frame) -> bool:
    return current_voice == "News Lady"


async def british_lady_filter(frame) -> bool:
    return current_voice == "British Lady"


async def barbershop_man_filter(frame) -> bool:
    return current_voice == "Barbershop Man"


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Pipecat",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        news_lady = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="bf991597-6c13-47e4-8411-91ec2de5c466",  # Newslady
        )

        british_lady = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        barbershop_man = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Barbershop Man
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        llm.register_function("switch_voice", switch_voice)

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "switch_voice",
                    "description": "Switch your voice only when the user asks you to",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "voice": {
                                "type": "string",
                                "description": "The voice the user wants you to use",
                            },
                        },
                        "required": ["voice"],
                    },
                },
            )
        ]
        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities. Respond to what the user said in a creative and helpful way. Your output should not include non-alphanumeric characters. You can do the following voices: 'News Lady', 'British Lady' and 'Barbershop Man'.",
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                context_aggregator.user(),  # User responses
                llm,  # LLM
                ParallelPipeline(  # TTS (one of the following vocies)
                    [FunctionFilter(news_lady_filter), news_lady],  # News Lady voice
                    [FunctionFilter(british_lady_filter), british_lady],  # British Lady voice
                    [FunctionFilter(barbershop_man_filter), barbershop_man],  # Barbershop Man voice
                ),
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": f"Please introduce yourself to the user and let them know the voices you can do. Your initial responses should be as if you were a {current_voice}.",
                }
            )
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
