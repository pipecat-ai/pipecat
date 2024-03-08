import aiohttp
import asyncio
import json
import random
import logging
import os
import re
import wave
from typing import AsyncGenerator
from PIL import Image

from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.pipeline.aggregators import (
    LLMAssistantContextAggregator,
    LLMContextAggregator,
    LLMUserContextAggregator,
    UserResponseAggregator,
    LLMResponseAggregator,
)
from examples.support.runner import configure
from dailyai.pipeline.frames import (
    LLMMessagesQueueFrame,
    TranscriptionQueueFrame,
    Frame,
    TextFrame,
    LLMFunctionCallFrame,
    LLMFunctionStartFrame,
    LLMResponseEndFrame,
    StartFrame,
    AudioFrame,
    SpriteFrame,
    ImageFrame,
)
from dailyai.services.ai_services import FrameLogger, AIService

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

sounds = {}
sound_files = ["clack-short.wav", "clack.wav", "clack-short-quiet.wav"]

script_dir = os.path.dirname(__file__)

for file in sound_files:
    # Build the full path to the image file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the image and convert it to bytes
    with wave.open(full_path) as audio_file:
        sounds[file] = audio_file.readframes(-1)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        global transport
        global llm
        global tts

        transport = DailyTransportService(
            room_url,
            token,
            "Story Cat",
            5,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False,
            start_transcription=True,
            vad_enabled=True,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a storytelling cat who loves to make up fantastic, fun, and educational stories for children between the ages of 5 and 10 years old. Your stories are full of friendly, magical creatures. Your stories are never scary. Each sentence of your story will become a page in a storybook. Stop after 3-4 sentences and give the child a choice to make that will influence the next part of the story. Once the child responds, start by saying something nice about the choice they made, then include [start] in your response. Include [break] after each sentence of the story. Include [prompt] between the story and the prompt.",
            }
        ]

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_CHATGPT_API_KEY"),
            model="gpt-4-1106-preview",
        )  # gpt-4-1106-preview
        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="jBpfuIE2acCO8z3wKNLl",
        )  # matilda
        lra = LLMResponseAggregator(messages)
        ura = UserResponseAggregator(messages)

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            # We're being a bit tricky here by using a special system prompt to
            # ask the user for a story topic. After their intial response, we'll
            # use a different system prompt to create story pages.
            intro_messages = [
                {
                    "role": "system",
                    "content": "You are a storytelling cat who loves to make up fantastic, fun, and educational stories for children between the ages of 5 and 10 years old. Your stories are full of friendly, magical creatures. Your stories are never scary. Begin by asking what a child wants you to tell a story about. Keep your reponse to only a few sentences.",
                }
            ]
            lca = LLMAssistantContextAggregator(messages)
            await tts.run_to_queue(
                transport.send_queue,
                lca.run(
                    llm.run([LLMMessagesQueueFrame(intro_messages)]),
                ),
            )

        async def storytime():
            pipeline = Pipeline(processors=[ura, llm, tts, lra])
            await transport.run_uninterruptible_pipeline(
                pipeline,
            )

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True
        try:
            await asyncio.gather(transport.run(), storytime())
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("whoops")
            transport.stop()


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
