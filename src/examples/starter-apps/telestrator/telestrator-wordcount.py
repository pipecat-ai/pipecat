import asyncio
import aiohttp
import logging
import os
import random
from typing import AsyncGenerator

from dailyai.pipeline.frames import Frame, LLMMessagesQueueFrame, RequestVideoImageFrame, LLMResponseEndFrame, TelestratorImageFrame, ImageFrame, TextFrame
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

narrators = [{"voice_id": "wDRBdcyPzQOCeq51IxW5",
              "prompt": "Describe the image in nine words."},
             {"voice_id": "M3bAX0o3Ptb2l6XqwQJV",
              "prompt": "Describe the image in one sentence, in the style of John Oliver's Last Week Tonight show."},
             {"voice_id": "lJm5d2ZZ3UE4qYOxl2t7",
              "prompt": "Describe the image in one sentence, in the style of Oprah Winfrey."},
             {"voice_id": "7SNUlQ8GAbnZxRO9CKOt",
              "prompt": "Describe the image in one sentence, in the style of a royal pronouncement by the Queen of England."},
             {"voice_id": "gvpBhHjzfd7M2WedYVUI",
              "prompt": "Describe the image in one sentence, in the style of Captain Picard from Star Trek."},
             {"voice_id": "bnyr1EF3snReVXauGBNn",
              "prompt": "Describe the image in one sentence, in the style of Maya Angelou."}]

# random.shuffle(narrators)
print(f"$$$ narrators: {narrators}")
narrator = {"narrator": narrators[0]}


class TranslationProcessor(FrameProcessor):
    def __init__(self, in_language, out_language):
        self._in_language = in_language
        self._out_language = out_language

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, TextFrame):
            context = [
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence in {self._in_language}, and your task is to translate it into {self._out_language}.",
                },
                {"role": "user", "content": frame.text},
            ]

            yield LLMMessagesQueueFrame(context)
        else:
            yield frame


class NarratorShuffle(FrameProcessor):
    def __init__(self, narrator, narrators):
        self._narrator = narrator
        self._narrators = narrators
        self._i = 0

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, (ImageFrame, TelestratorImageFrame)):
            self._i += 1
            if self._i >= len(self._narrators):
                print(f"### shuffling narrators")
                random.shuffle(self._narrators)
                self._i = 0

            self._narrator["narrator"] = self._narrators[self._i]
            print(f"### new narrator is {self._narrator}")
        yield frame


class VideoImageFrameProcessor(FrameProcessor):
    def __init__(self, narrator):
        self._narrator = narrator

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, (VideoImageFrame, TelestratorImageFrame)):
            yield VisionFrame(self._narrator["narrator"]["prompt"], frame.image)
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
            narrator=narrator,
            aggregate_sentences=False
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_CHATGPT_API_KEY"),
            model="gpt-4-turbo-preview")

        vs = OpenAIVisionService(api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))
        vifp = VideoImageFrameProcessor(narrator)
        ir = ImageRefresher()
        img = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )
        tiw = TelestratorImageWrapper()
        lfra = LLMFullResponseAggregator()
        lfra1 = LLMFullResponseAggregator()
        lfra2 = LLMFullResponseAggregator()
        lfra3 = LLMFullResponseAggregator()
        lfra4 = LLMFullResponseAggregator()
        fl0 = FrameLogger("@@@ About to describe")
        fl1 = FrameLogger("!!! About to image gen")
        f4 = FrameLogger("((( partway through )))")
        f5 = FrameLogger("!!! f5")
        ns = NarratorShuffle(narrator, narrators)
        t1 = TranslationProcessor("English", "Spanish")
        t2 = TranslationProcessor("Spanish", "German")
        t3 = TranslationProcessor("German", "Japanese")
        t4 = TranslationProcessor("Japanese", "English")
        pipeline = Pipeline(
            processors=[
                fl0,
                vifp,
                vs,
                lfra,
                tts,
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
