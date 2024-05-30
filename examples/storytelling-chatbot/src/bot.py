import argparse
import asyncio
import aiohttp
import os
import sys


from pipecat.frames.frames import LLMMessagesFrame, StopTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.fal import FalImageGenService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyTransportMessageFrame

from processors import StoryProcessor, StoryImageProcessor
from prompts import LLM_BASE_PROMPT, LLM_INTRO_PROMPT, CUE_USER_TURN
from utils.helpers import load_sounds, load_images

from loguru import logger

from dotenv import load_dotenv
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
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=768,
                camera_out_height=768,
                transcription_enabled=True,
                vad_enabled=True,
            )
        )

        logger.debug("Transport created for room:" + room_url)

        # -------------- Services --------------- #

        llm_service = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o"
        )

        tts_service = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        fal_service_params = FalImageGenService.InputParams(
            image_size={
                "width": 768,
                "height": 768
            }
        )

        fal_service = FalImageGenService(
            aiohttp_session=session,
            model="fal-ai/fast-lightning-sdxl",
            params=fal_service_params,
            key=os.getenv("FAL_KEY"),
        )

        # --------------- Setup ----------------- #

        message_history = [LLM_BASE_PROMPT]
        story_pages = []

        # We need aggregators to keep track of user and LLM responses
        llm_responses = LLMAssistantResponseAggregator(message_history)
        user_responses = LLMUserResponseAggregator(message_history)

        # -------------- Processors ------------- #

        story_processor = StoryProcessor(message_history, story_pages)
        image_processor = StoryImageProcessor(fal_service)

        # -------------- Story Loop ------------- #

        runner = PipelineRunner()

        # The intro pipeline is used to start
        # the story (as per LLM_INTRO_PROMPT)
        intro_pipeline = Pipeline([llm_service, tts_service, transport.output()])

        intro_task = PipelineTask(intro_pipeline)

        logger.debug("Waiting for participant...")

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.debug("Participant joined, storytime commence!")
            transport.capture_participant_transcription(participant["id"])
            await intro_task.queue_frames(
                [
                    images['book1'],
                    LLMMessagesFrame([LLM_INTRO_PROMPT]),
                    DailyTransportMessageFrame(CUE_USER_TURN),
                    sounds["listening"],
                    images['book2'],
                    StopTaskFrame()
                ]
            )

        # We run the intro pipeline. This will start the transport. The intro
        # task will exit after StopTaskFrame is processed.
        await runner.run(intro_task)

        # The main story pipeline is used to continue the story based on user
        # input.
        main_pipeline = Pipeline([
            transport.input(),
            user_responses,
            llm_service,
            story_processor,
            image_processor,
            tts_service,
            transport.output(),
            llm_responses
        ])

        main_task = PipelineTask(main_pipeline)

        await runner.run(main_task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Storyteller Bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t))
