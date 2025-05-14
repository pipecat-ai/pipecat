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
from processors import StoryImageProcessor, StoryProcessor
from prompts import CUE_USER_TURN, LLM_BASE_PROMPT
from utils.helpers import load_images, load_sounds

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.image import GoogleImageGenService
from pipecat.services.google.llm import GoogleLLMService
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


async def main(room_url, token=None):
    async with aiohttp.ClientSession() as session:
        # -------------- Transport --------------- #

        transport = DailyTransport(
            room_url,
            token,
            "Storytelling Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_width=1024,
                video_out_height=1024,
                transcription_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
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
        main_pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm_service,
                story_processor,
                image_processor,
                tts_service,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        main_task = PipelineTask(
            main_pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
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
