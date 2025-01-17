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
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Example callback using the new style with retry control
async def handle_user_idle(processor: UserIdleProcessor, retry_count: int) -> bool:
    if retry_count == 1:
        # First attempt: Add a gentle prompt to the conversation
        messages.append(
            {
                "role": "system",
                "content": "The user has been quiet for a while. Politely ask if they're still there.",
            }
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        return True
    elif retry_count == 2:
        # Second attempt: More direct prompt
        messages.append(
            {
                "role": "system",
                "content": "The user is still inactive. Ask if they would like to continue the conversation.",
            }
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        return True
    else:
        # Third attempt: End the conversation
        messages.append(
            {
                "role": "system",
                "content": "The user has been inactive for too long. Politely end the conversation.",
            }
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        await task.queue_frame(EndFrame())
        return False


async def main():
    global task, messages, context_aggregator  # Make these accessible to the idle handler

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

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Create the idle processor
        idle_processor = UserIdleProcessor(
            callback=handle_user_idle,
            timeout=5.0,  # 5 seconds of inactivity before triggering
        )

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                idle_processor,  # Add the idle processor
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
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
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
