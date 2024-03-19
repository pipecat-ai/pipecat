import asyncio
import aiohttp
import logging
import os
from typing import AsyncGenerator

from dailyai.pipeline.frames import Frame, LLMMessagesQueueFrame, RequestVideoImageFrame, LLMResponseEndFrame, TelestratorImageFrame, ImageFrame
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
    LLMFullResponseAggregator
)
from dailyai.pipeline.frames import VideoImageFrame, VisionFrame
from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


class VideoImageFrameProcessor(FrameProcessor):
    def __init__(self):
        pass

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, VideoImageFrame) or isinstance(frame, TelestratorImageFrame):
            yield VisionFrame("Describe the image in one sentence, in the style of David Attenborough.", frame.image)
        else:
            yield frame


class ImageRefresher(FrameProcessor):
    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, LLMResponseEndFrame):
            yield RequestVideoImageFrame(participantId=None)
            yield frame
        else:
            yield frame


class TelestratorImageWrapper(FrameProcessor):
    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, ImageFrame):
            yield TelestratorImageFrame(None, frame.image)
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
            vad_enabled=False,
            receive_video=True,
            receive_video_fps=0
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
        vifp = VideoImageFrameProcessor()
        ir = ImageRefresher()
        img = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )
        tiw = TelestratorImageWrapper()
        lfra = LLMFullResponseAggregator()
        fl1 = FrameLogger("!!! About to image gen")
        pipeline = Pipeline(
            processors=[
                vifp,
                vs,
                tts,
                lfra,
                fl1,
                img,
                tiw,
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
