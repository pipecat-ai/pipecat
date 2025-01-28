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
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartInterruptionFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

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
        # Convert timestamp to seconds for readability
        time_sec = timestamp / 1_000_000_000

        # Create direction arrow
        arrow = "→" if direction == FrameDirection.DOWNSTREAM else "←"

        if isinstance(frame, StartInterruptionFrame):
            logger.info(f"⚡ INTERRUPTION START: {src} {arrow} {dst} at {time_sec:.2f}s")
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info(f"🤖 BOT START SPEAKING: {src} {arrow} {dst} at {time_sec:.2f}s")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info(f"🤖 BOT STOP SPEAKING: {src} {arrow} {dst} at {time_sec:.2f}s")


class LLMLogObserver(BaseObserver):
    """Observer to log LLM activity to the console.

    Logs all frame instances of:
    - LLMFullResponseStartFrame (only from LLM service)
    - LLMTextFrame
    - LLMFullResponseEndFrame (only from LLM service)

    This allows you to track when the LLM starts responding, what it generates, and when it finishes.
    Log format: [LLM EVENT]: [details] at [timestamp]s
    """

    async def on_push_frame(
        self,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
        timestamp: int,
    ):
        time_sec = timestamp / 1_000_000_000

        # Only log start/end frames from OpenAILLMService
        if isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
            if isinstance(src, OpenAILLMService):
                event = "START" if isinstance(frame, LLMFullResponseStartFrame) else "END"
                logger.info(f"🧠 LLM {event} RESPONSE at {time_sec:.2f}s")
        # Log all LLMTextFrames
        elif isinstance(frame, LLMTextFrame):
            logger.info(f"🧠 LLM GENERATING: {frame.text!r} at {time_sec:.2f}s")


async def main():
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

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
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
                observers=[DebugObserver(), LLMLogObserver()],
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
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
