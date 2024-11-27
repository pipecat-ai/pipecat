#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import time

import aiohttp
from loguru import logger
from runner import configure

from pipecat.frames.frames import (
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url, None, "Say One Thing", DailyParams(audio_out_enabled=True)
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        runner = PipelineRunner()

        task = PipelineTask(
            Pipeline(
                [
                    context_aggregator.user(),
                    llm,
                    tts,
                    transport.output(),
                    context_aggregator.assistant(),
                ]
            )
        )

        # Register an event handler so we can play the audio when the
        # participant joins.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Create frames for 3 seconds
            start_time = time.time()
            while time.time() - start_time < 300:
                timestamp = time.time()
                frames = [
                    UserStartedSpeakingFrame(),
                    TranscriptionFrame("Tell a joke about dogs.", "user_id", timestamp),
                    UserStoppedSpeakingFrame(),
                ]
                await task.queue_frames(frames)
                await asyncio.sleep(5)  # Small delay between frame sets
                next_frames = [
                    StartInterruptionFrame(),
                    TranscriptionFrame("Tell a joke about cats.", "user_id", timestamp),
                    StopInterruptionFrame(),
                ]
                await task.queue_frames(next_frames)

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
