import asyncio

import aiohttp
import logging
import os
from PIL import Image
from typing import AsyncGenerator

from dailyai.pipeline.aggregators import (
    LLMUserResponseAggregator,
    ParallelPipeline,
    VisionImageFrameAggregator,
    SentenceAggregator
)
from dailyai.pipeline.frames import (
    ImageFrame,
    SpriteFrame,
    Frame,
    LLMMessagesFrame,
    AudioFrame,
    PipelineStartedFrame,
    TTSEndFrame,
    TextFrame,
    UserImageFrame,
    UserImageRequestFrame,
)
from dailyai.services.moondream_ai_service import MoondreamService
from dailyai.pipeline.pipeline import FrameProcessor, Pipeline
from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

user_request_answer = "Let me take a look."

sprites = []

script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(img.tobytes())

flipped = sprites[::-1]
sprites.extend(flipped)

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = ImageFrame(sprites[0], (1024, 576))
talking_frame = SpriteFrame(images=sprites)


class TalkingAnimation(FrameProcessor):
    """
    This class starts a talking animation when it receives an first AudioFrame,
    and then returns to a "quiet" sprite when it sees a LLMResponseEndFrame.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, AudioFrame):
            if not self._is_talking:
                yield talking_frame
                yield frame
                self._is_talking = True
            else:
                yield frame
        elif isinstance(frame, TTSEndFrame):
            yield quiet_frame
            yield frame
            self._is_talking = False
        else:
            yield frame


class AnimationInitializer(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PipelineStartedFrame):
            yield quiet_frame
            yield frame
        else:
            yield frame


class UserImageRequester(FrameProcessor):
    participant_id: str | None

    def __init__(self):
        super().__init__()
        self.participant_id = None

    def set_participant_id(self, participant_id: str):
        self.participant_id = participant_id

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if self.participant_id and isinstance(frame, TextFrame):
            if frame.text == user_request_answer:
                yield UserImageRequestFrame(self.participant_id)
                yield TextFrame("Describe the image in a short sentence.")
        elif isinstance(frame, UserImageFrame):
            yield frame


class TextFilterProcessor(FrameProcessor):
    text: str

    def __init__(self, text: str):
        self.text = text

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, TextFrame):
            if frame.text != self.text:
                yield frame
        else:
            yield frame


class ImageFilterProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if not isinstance(frame, ImageFrame):
            yield frame


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=True,
            camera_width=1024,
            camera_height=576,
            vad_enabled=True,
            video_rendering_enabled=True
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="pNInz6obpgDQGcFmaJgB",
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview")

        ta = TalkingAnimation()
        ai = AnimationInitializer()

        sa = SentenceAggregator()
        ir = UserImageRequester()
        va = VisionImageFrameAggregator()
        # If you run into weird description, try with use_cpu=True
        moondream = MoondreamService()

        tf = TextFilterProcessor(user_request_answer)
        imgf = ImageFilterProcessor()

        messages = [
            {
                "role": "system",
                "content": f"You are Chatbot, a friendly, helpful robot. Let the user know that you are capable of chatting or describing what you see. Your goal is to demonstrate your capabilities in a succinct way. Reply with only '{user_request_answer}' if the user asks you to describe what you see. Your output will be converted to audio so never include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",
            },
        ]

        ura = LLMUserResponseAggregator(messages)

        pipeline = Pipeline([
            ai, ura, llm, ParallelPipeline(
                [[sa, ir, va, moondream], [tf, imgf]]
            ),
            tts, ta
        ])

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport, participant):
            transport.render_participant_video(participant["id"], framerate=0)
            ir.set_participant_id(participant["id"])
            await pipeline.queue_frames([LLMMessagesFrame(messages)])

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True

        await asyncio.gather(transport.run(pipeline))


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
