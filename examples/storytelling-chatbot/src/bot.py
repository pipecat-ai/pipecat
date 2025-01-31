#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from processors import StoryImageFrame, StoryImageProcessor, StoryPageFrame, StoryProcessor
from prompts import CUE_USER_TURN, LLM_BASE_PROMPT
from utils.helpers import load_images, load_sounds

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    ImageRawFrame,
    MetadataFrame,
    SystemFrame,
    TextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.sync_parallel_pipeline import SyncParallelPipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.logger import FrameLogger
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.fal import FalImageGenService
from pipecat.services.google import GoogleImageGenService, GoogleLLMService
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
    DailyTransportMessageFrame,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sounds = load_sounds(["listening.wav"])
images = load_images(["book1.png", "book2.png"])


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

        if isinstance(frame, ImageRawFrame):
            logger.info(
                f"⚡ RAW IMAGE FRAME: {src} {arrow} {dst} at {time_sec:.2f}s, metadata: {frame.metadata}"
            )
        elif isinstance(frame, StoryPageFrame):
            logger.info(
                f"⚡ STORY PAGE FRAME: {src} {arrow} {dst} at {time_sec:.2f}s, metadata: {frame.metadata}"
            )
        elif isinstance(frame, StoryImageFrame):
            logger.info(
                f"⚡ STORY IMAGE FRAME: {src} {arrow} {dst} at {time_sec:.2f}s, metadata: {frame.metadata}"
            )
        elif isinstance(frame, TextFrame):
            logger.info(
                f"⚡ TEXT FRAME: {src} {arrow} {dst} at {time_sec:.2f}s, metadata: {frame.metadata}"
            )
        elif isinstance(frame, TTSStartedFrame):
            logger.info(
                f"⚡ TTS STARTED FRAME: {src} {arrow} {dst} at {time_sec:.2f}s, metadata: {frame.metadata}"
            )
        elif isinstance(frame, TTSStoppedFrame):
            logger.info(
                f"⚡ TTS STOPPED FRAME: {src} {arrow} {dst} at {time_sec:.2f}s, metadata: {frame.metadata}"
            )
        elif isinstance(frame, MetadataFrame):
            logger.info(
                f"⚡ METADATA FRAME: {src} {arrow} {dst} at {time_sec:.2f}s, metadata: {frame.metadata}"
            )


async def main(room_url, token=None):
    async with aiohttp.ClientSession() as session:
        # -------------- Transport --------------- #

        transport = DailyTransport(
            room_url,
            token,
            "Storytelling Bot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1024,
                transcription_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_enabled=True,
            ),
        )

        logger.debug("Transport created for room:" + room_url)

        # -------------- Services --------------- #

        llm_service = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

        tts_service = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=os.getenv("ELEVENLABS_VOICE_ID")
        )

        image_gen = GoogleImageGenService(api_key=os.getenv("GOOGLE_API_KEY"))

        # --------------- Setup ----------------- #

        message_history = [LLM_BASE_PROMPT]
        story_pages = []

        # We need aggregators to keep track of user and LLM responses
        context = OpenAILLMContext(message_history)
        context_aggregator = llm_service.create_context_aggregator(context)

        # -------------- Processors ------------- #

        story_processor = StoryProcessor(message_history, story_pages)
        image_processor = StoryImageProcessor(image_gen)

        # -------------- Story Loop ------------- #

        runner = PipelineRunner()

        logger.debug("Waiting for participant...")
        after = FrameLogger("After", "red", ignored_frame_types=[SystemFrame, AudioRawFrame])
        before = FrameLogger("Before", "cyan", ignored_frame_types=[SystemFrame, AudioRawFrame])
        main_pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm_service,
                story_processor,
                # SyncParallelPipeline([image_processor], [tts_service]),
                before,
                image_processor,
                after,
                tts_service,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        main_task = PipelineTask(
            main_pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                observers=[DebugObserver()],
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.debug("Participant joined, storytime commence!")
            await transport.capture_participant_transcription(participant["id"])
            await main_task.queue_frames(
                [
                    images["book1"],
                    context_aggregator.user().get_context_frame(),
                    DailyTransportMessageFrame(CUE_USER_TURN),
                    # sounds["listening"],
                    images["book2"],
                ]
            )

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await main_task.cancel()

        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                # Here we don't want to cancel, we just want to finish sending
                # whatever is queued, so we use an EndFrame().
                await main_task.queue_frame(EndFrame())

        await runner.run(main_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Storyteller Bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t))
