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
    BotSpeakingFrame,
    EndFrame,
    Frame,
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class DebugProcessor(FrameProcessor):
    def __init__(self, name, **kwargs):
        self._name = name
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not (
            isinstance(frame, BotSpeakingFrame)
        ):
            logger.debug(f"--- {self._name}: {frame} {direction}")
        await self.push_frame(frame, direction)


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

        dp = DebugProcessor("dp")

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        runner = PipelineRunner()

        task = PipelineTask(
            Pipeline(
                [
                    dp,
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
            participant_id = participant.get("info", {}).get("participantId", "")

            await task.queue_frames(
                [
                    UserStartedSpeakingFrame(),
                    TranscriptionFrame("Tell a joke about dogs.", participant_id, time.time()),
                    UserStoppedSpeakingFrame(),
                ]
            )
            # await asyncio.sleep(5)  # Small delay between frame sets

            # Create frames for 60 seconds
            start_time = time.time()
            while time.time() - start_time < 30:
                elapsed_time = round(time.time() - start_time)
                logger.info(f"Running for {elapsed_time} seconds")
                await asyncio.sleep(5)  # Small delay between frame sets
                await task.queue_frames(
                    [
                        StartInterruptionFrame(),
                        TranscriptionFrame("Tell a joke about cats.", participant_id, time.time()),
                        StopInterruptionFrame(),
                    ]
                )
                await asyncio.sleep(5)  # Small delay between frame sets
                await task.queue_frames(
                    [
                        StartInterruptionFrame(),
                        TranscriptionFrame("Tell a joke about dogs.", participant_id, time.time()),
                        StopInterruptionFrame(),
                    ]
                )
            await task.queue_frame(EndFrame())

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
