#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Bot Implementation.

This module implements a chatbot using OpenAI's GPT-4 model for natural language
processing. It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Text-to-speech using ElevenLabs
- Support for both English and Spanish

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow.
"""

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    ActionResult,
    RTVIAction,
    RTVIActionArgument,
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

script_dir = os.path.dirname(__file__)


async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily transport
    - Speech-to-text and text-to-speech services
    - Language model integration
    - RTVI event handling
    """
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Set up Daily transport with video/audio parameters
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=False,
                transcription_enabled=True,
            ),
        )

        # Initialize LLM service
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",
            },
        ]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        async def action_llm_append_to_messages_handler(
            rtvi: RTVIProcessor, service: str, arguments: dict[str, any]
        ) -> ActionResult:
            run_immediately = (
                arguments["run_immediately"] if "run_immediately" in arguments else True
            )

            if run_immediately:
                await rtvi.interrupt_bot()

            # We just interrupted the bot so it should be fine to use the
            # context directly instead of through frame.
            if "messages" in arguments and arguments["messages"]:
                mess = arguments["messages"]
                frame = LLMMessagesAppendFrame(messages=arguments["messages"])
                await rtvi.push_frame(frame)

            if run_immediately:
                frame = context_aggregator.user().get_context_frame()
                await rtvi.push_frame(frame)

            return True

        action_llm_append_to_messages = RTVIAction(
            service="llm",
            action="append_to_messages",
            result="bool",
            arguments=[
                RTVIActionArgument(name="messages", type="array"),
                RTVIActionArgument(name="run_immediately", type="bool"),
            ],
            handler=action_llm_append_to_messages_handler,
        )

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
        rtvi.register_action(action_llm_append_to_messages)

        #
        # RTVI events for Pipecat client UI
        #

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                context_aggregator.user(),
                llm,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
