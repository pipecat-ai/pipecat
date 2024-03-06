import argparse
import asyncio
import os
from typing import AsyncGenerator
import aiohttp
import requests
import time
import urllib.parse

from PIL import Image
from dailyai.pipeline.frames import ImageFrame, Frame

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.ai_services import AIService
from dailyai.pipeline.aggregators import LLMAssistantContextAggregator, LLMUserContextAggregator
from dailyai.services.fal_ai_services import FalImageGenService

from examples.foundational.support.runner import configure


class ImageSyncAggregator(AIService):
    def __init__(self, speaking_path: str, waiting_path: str):
        self._speaking_image = Image.open(speaking_path)
        self._speaking_image_bytes = self._speaking_image.tobytes()

        self._waiting_image = Image.open(waiting_path)
        self._waiting_image_bytes = self._waiting_image.tobytes()

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        yield ImageFrame(None, self._speaking_image_bytes)
        yield frame
        yield ImageFrame(None, self._waiting_image_bytes)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
            room_url,
            token,
            "Respond bot",
            5,
        )
        transport._camera_enabled = True
        transport._camera_width = 1024
        transport._camera_height = 1024
        transport._mic_enabled = True
        transport._mic_sample_rate = 16000

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"))
        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"))
        img = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"))

        async def get_images():
            get_speaking_task = asyncio.create_task(
                img.run_image_gen("An image of a cat speaking")
            )
            get_waiting_task = asyncio.create_task(
                img.run_image_gen("An image of a cat waiting")
            )

            (speaking_data, waiting_data) = await asyncio.gather(
                get_speaking_task, get_waiting_task
            )

            return speaking_data, waiting_data

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await tts.say("Hi, I'm listening!", transport.send_queue)

        async def handle_transcriptions():
            messages = [
                {"role": "system", "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way."},
            ]

            tma_in = LLMUserContextAggregator(
                messages, transport._my_participant_id
            )
            tma_out = LLMAssistantContextAggregator(
                messages, transport._my_participant_id
            )
            image_sync_aggregator = ImageSyncAggregator(
                os.path.join(os.path.dirname(__file__), "assets", "speaking.png"),
                os.path.join(os.path.dirname(__file__), "assets", "waiting.png"),
            )
            await tts.run_to_queue(
                transport.send_queue,
                image_sync_aggregator.run(
                    tma_out.run(
                        llm.run(
                            tma_in.run(
                                transport.get_receive_frames()
                            )
                        )
                    )
                )
            )

        transport.transcription_settings["extra"]["punctuate"] = True
        await asyncio.gather(transport.run(), handle_transcriptions())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
