import asyncio
import os
import logging
from typing import AsyncGenerator
import aiohttp
from PIL import Image

from dailyai.pipeline.frames import ImageFrame, Frame, TextFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.ai_services import AIService
from dailyai.pipeline.aggregators import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


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
        transport = DailyTransport(
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
        transport.transcription_settings["extra"]["punctuate"] = True

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so it should not include any special characters. Respond to what the user said in a creative and helpful way.",
            },
        ]

        tma_in = LLMUserContextAggregator(
            messages, transport._my_participant_id)
        tma_out = LLMAssistantContextAggregator(
            messages, transport._my_participant_id
        )
        image_sync_aggregator = ImageSyncAggregator(
            os.path.join(os.path.dirname(__file__), "assets", "speaking.png"),
            os.path.join(os.path.dirname(__file__), "assets", "waiting.png"),
        )

        pipeline = Pipeline([image_sync_aggregator, tma_in, llm, tma_out, tts])

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await pipeline.queue_frames([TextFrame("Hi, I'm listening!")])

        await transport.run(pipeline)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
