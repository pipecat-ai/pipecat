#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import math
import os
import random
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.frames.frames import BotSpeakingFrame, EndFrame, Frame, TextFrame, TTSSpeakFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyOutputTransport, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class DebugObserver(BaseObserver):
    """Observer to log interruptions and bot speaking events to the console.

    Logs all frame instances of:
    - StartInterruptionFrame
    - BotStartedSpeakingFrame
    - BotStoppedSpeakingFrame

    This allows you to see the frame flow from processor to processor through the pipeline for these frames.
    Log format: [EVENT TYPE]: [source processor] → [destination processor] at [timestamp]s
    """

    async def on_push_frame(
        self,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
        timestamp: int,
    ):
        arrow = "→" if direction == FrameDirection.DOWNSTREAM else "←"
        # Convert timestamp to seconds for readability
        time_sec = timestamp / 1_000_000_000

        if isinstance(frame, BotSpeakingFrame):
            return

        if isinstance(dst, DailyOutputTransport):
            logger.debug(f"{frame} {src} {arrow} {dst} at {time_sec:.2f}s")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport1 = DailyTransport(
            "https://hush.daily.co/sip",
            None,
            "Don't Do Anything",
            DailyParams(audio_out_enabled=True),
        )

        transport2 = DailyTransport(
            "https://hush.daily.co/demo",
            None,
            "Summarize Call",
            DailyParams(audio_out_enabled=True),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        runner = PipelineRunner()

        async def true_filter(frame) -> bool:
            return True

        async def false_filter(frame) -> bool:
            return False

        pipeline = Pipeline(
            [
                transport1.input(),
                transport2.input(),
                ParallelPipeline(
                    [transport1.output()],
                    [tts, transport2.output()],
                ),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
                observers=[DebugObserver()],
            ),
        )

        # Register an event handler so we can play the audio when the
        # participant joins.
        @transport1.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            participant_name = participant.get("info", {}).get("userName", "")
            logger.info(f"-- {participant_name} joined transport1")

        def get_call_summary():
            """In a real app this would be a call to a database or API."""
            # Randomly choose between two options
            message = random.choice(
                [
                    "Alice needs help finding her customer record.",
                    "Bob is calling about his lost password.",
                ]
            )

            return message

        @transport2.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            participant_name = participant.get("info", {}).get("userName", "")
            logger.info(f"-- {participant_name} joined transport2")
            call_summary = get_call_summary()
            await task.queue_frames(
                [
                    TTSSpeakFrame(
                        f"Hi {participant_name}! Here's the summary of the call: {call_summary}"
                    ),
                    EndFrame(),
                ]
            )

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
