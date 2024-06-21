#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


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
                vad_analyzer=SileroVADAnalyzer()
            )
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_name=sys.argv[1] if len(sys.argv) > 1 else "British Lady"
        )

        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20240620",
            temperature=1.0
        )

        # todo: think more about how to handle system prompts in a more general way. OpenAI,
        # Google, and Anthropic all have slightly different approaches to providing a system
        # prompt.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are participating in a friendly competition to invent creative "
                    "new ice cream flavors. Say the craziest flavor you can think of "
                    "then wait for your opponent to come up with a different crazy flavor. "
                    "then respond with another flavor idea. Repeat forever. Say only the "
                    "ice cream flavors and nothing else. End each ice cream flavor statement "
                    "with an exclamation mark! Go ..."
                )
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline([
            transport.input(),   # Transport user input
            tma_in,              # User responses
            llm,                 # LLM
            tts,                 # TTS
            transport.output(),  # Transport bot output
            tma_out,             # Assistant spoken responses
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True, enable_metrics=True))

        # When a participant joins, start transcription for that participant so the
        # bot can "hear" and respond to them.
        @ transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])

        # When the first participant joins, the bot should introduce itself.
        @ transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))

# '{"action":"app-message","data":{"metrics":{"ttfb":[{"name":"AnthropicLLMService#0","time":0.5975627899169922}]},"type":"pipecat-metrics"},"fromId":"592d3489-90ba-401d-a760-c1a863d64a4a","callFrameId":"17189290998160.035120590426112264"}'
# [Durian and Limburger Cheese Charcoal Activated Tar Twist!]
# [Fermented Fish Sauce and Ghost Pepper Bubblegum Cotton Candy Nightmare!]
# [Spoiled Yogurt and Ghost Pepper Gummy Bear Blizzard!]
