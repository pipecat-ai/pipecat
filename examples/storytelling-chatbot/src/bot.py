import asyncio
import aiohttp
import logging
import os
import argparse

from dailyai.pipeline.pipeline import Pipeline
from dailyai.pipeline.frames import (
    AudioFrame,
    ImageFrame,
    EndPipeFrame,
    LLMMessagesFrame,
    SendAppMessageFrame
)
from dailyai.pipeline.aggregators import (
    LLMUserResponseAggregator,
    LLMAssistantResponseAggregator,
)
from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.fal_ai_services import FalImageGenService

from processors import StoryProcessor, StoryImageProcessor
from prompts import LLM_BASE_PROMPT, LLM_INTRO_PROMPT, CUE_USER_TURN
from utils.helpers import load_sounds, load_images

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"[STORYBOT] %(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.INFO)


sounds = load_sounds(["listening.wav"])
images = load_images(["book1.png", "book2.png"])


async def main(room_url, token=None):
    async with aiohttp.ClientSession() as session:

        # -------------- Transport --------------- #

        transport = DailyTransport(
            room_url,
            token,
            "Storytelling Bot",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            vad_enabled=True,
            camera_framerate=30,
            camera_bitrate=680000,
            camera_enabled=True,
            camera_width=768,
            camera_height=768,
        )

        logger.debug("Transport created for room:" + room_url)

        # -------------- Services --------------- #

        llm_service = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo"
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

        logger.debug("Waiting for participant...")

        start_storytime_event = asyncio.Event()

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport, participant):
            logger.debug("Participant joined, storytime commence!")
            start_storytime_event.set()

        # The storytime coroutine will wait for the start_storytime_event
        # to be set before starting the storytime pipeline
        async def storytime():
            await start_storytime_event.wait()

            # The intro pipeline is used to start
            # the story (as per LLM_INTRO_PROMPT)
            intro_pipeline = Pipeline(processors=[
                llm_service,
                tts_service,
            ], sink=transport.send_queue)

            await intro_pipeline.queue_frames(
                [
                    ImageFrame(images['book1'], (768, 768)),
                    LLMMessagesFrame([LLM_INTRO_PROMPT]),
                    SendAppMessageFrame(CUE_USER_TURN, None),
                    AudioFrame(sounds["listening"]),
                    ImageFrame(images['book2'], (768, 768)),
                    EndPipeFrame(),
                ]
            )

            # We start the pipeline as soon as the user joins
            await intro_pipeline.run_pipeline()

            # The main story pipeline is used to continue the
            # story based on user input
            pipeline = Pipeline(processors=[
                user_responses,
                llm_service,
                story_processor,
                image_processor,
                tts_service,
                llm_responses,
            ])

            await transport.run_pipeline(pipeline)

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True

        try:
            await asyncio.gather(transport.run(), storytime())
        except (asyncio.CancelledError, KeyboardInterrupt):
            transport.stop()

        logger.debug("Pipeline finished. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily Storyteller Bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t))
