import asyncio
import aiohttp
import logging
import os

from typing import AsyncGenerator

from dailyai.pipeline.aggregators import FrameProcessor, UserResponseAggregator, VisionImageFrameAggregator

from dailyai.pipeline.frames import Frame, TextFrame, UserImageRequestFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.moondream_ai_service import MoondreamService
from dailyai.transports.daily_transport import DailyTransport

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


class UserImageRequester(FrameProcessor):
    participant_id: str

    def set_participant_id(self, participant_id: str):
        self.participant_id = participant_id

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if self.participant_id and isinstance(frame, TextFrame):
            yield UserImageRequestFrame(self.participant_id)
        yield frame


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Describe participant video",
            duration_minutes=5,
            mic_enabled=True,
            mic_sample_rate=16000,
            vad_enabled=True,
            start_transcription=True,
            video_rendering_enabled=True
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        user_response = UserResponseAggregator()

        image_requester = UserImageRequester()

        vision_aggregator = VisionImageFrameAggregator()

        moondream = MoondreamService()

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport, participant):
            await transport.say("Hi there! Feel free to ask me what I see.", tts)
            transport.render_participant_video(participant["id"], framerate=0)
            image_requester.set_participant_id(participant["id"])

        pipeline = Pipeline([user_response, image_requester, vision_aggregator, moondream, tts])

        await transport.run(pipeline)

if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
