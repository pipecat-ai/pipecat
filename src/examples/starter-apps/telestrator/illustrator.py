import asyncio
import aiohttp
import logging
import os
from typing import AsyncGenerator

from dailyai.pipeline.frames import Frame, LLMMessagesQueueFrame, RequestVideoImageFrame, LLMResponseEndFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame, TranscriptionQueueFrame, TextFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.pipeline.frame_processor import FrameProcessor
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.open_ai_services import OpenAILLMService, OpenAIVisionService
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.ai_services import FrameLogger
from dailyai.pipeline.aggregators import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from dailyai.pipeline.frames import VideoImageFrame, VisionFrame
from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


class VADAggregator(FrameProcessor):
    def __init__(self):
        self.aggregating = False
        self.aggregation = ""

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, UserStartedSpeakingFrame):
            self.aggregating = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.aggregating = False
            # Sometimes VAD triggers quickly on and off. If we don't get any transcription,
            # it creates empty LLM message queue frames
            if len(self.aggregation) > 0:
                yield TextFrame(self.aggregation)

                self.aggregation = ""
                yield frame
        elif isinstance(frame, TranscriptionQueueFrame) and self.aggregating:
            self.aggregation += f" {frame.text}"
        else:
            yield frame


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
            room_url,
            token,
            "Respond bot",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=True,
            camera_width=1024,
            camera_height=1024,
            vad_enabled=True,
            receive_video=True,
            receive_video_fps=0,
            vad_timeout_s=1.0
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_CHATGPT_API_KEY"),
            model="gpt-4-turbo-preview")

        vs = OpenAIVisionService(api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))
        vad = VADAggregator()
        img = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )
        fl = FrameLogger("!!! Start")
        fl2 = FrameLogger("!!! AFTER VAD")
        fl3 = FrameLogger("!!! After img")
        pipeline = Pipeline(
            processors=[
                fl,
                vad,
                fl2,
                img,
                fl3
            ],
        )

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await pipeline.queue_frames([RequestVideoImageFrame(participantId=None)])

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True
        await transport.run(pipeline)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
