import asyncio
import aiohttp
import os
import sys

from PIL import Image

from pipecat.frames.frames import (
    ImageRawFrame,
    SpriteFrame,
    Frame,
    LLMMessagesFrame,
    AudioRawFrame,
    TTSStoppedFrame,
    TextFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)

from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.moondream import MoondreamService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

user_request_answer = "Let me take a look."

sprites = []

script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(ImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

flipped = sprites[::-1]
sprites.extend(flipped)

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = sprites[0]
talking_frame = SpriteFrame(images=sprites)


class TalkingAnimation(FrameProcessor):
    """
    This class starts a talking animation when it receives an first AudioFrame,
    and then returns to a "quiet" sprite when it sees a TTSStoppedFrame.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False
        await self.push_frame(frame)


class UserImageRequester(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.participant_id = None

    def set_participant_id(self, participant_id: str):
        self.participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self.participant_id and isinstance(frame, TextFrame):
            if frame.text == user_request_answer:
                await self.push_frame(UserImageRequestFrame(self.participant_id), FrameDirection.UPSTREAM)
                await self.push_frame(TextFrame("Describe the image in a short sentence."))
        elif isinstance(frame, UserImageRawFrame):
            await self.push_frame(frame)


class TextFilterProcessor(FrameProcessor):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if frame.text != self.text:
                await self.push_frame(frame)
        else:
            await self.push_frame(frame)


class ImageFilterProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, ImageRawFrame):
            await self.push_frame(frame)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="pNInz6obpgDQGcFmaJgB",
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o")

        ta = TalkingAnimation()

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
                "content": f"You are Chatbot, a friendly, helpful robot. Let the user know that you are capable of chatting or describing what you see. Your goal is to demonstrate your capabilities in a succinct way. Reply with only '{user_request_answer}' if the user asks you to describe what you see. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",
            },
        ]

        ura = LLMUserResponseAggregator(messages)

        pipeline = Pipeline([
            transport.input(),
            ura,
            llm,
            ParallelPipeline(
                [sa, ir, va, moondream],
                [tf, imgf]),
            tts,
            ta,
            transport.output()
        ])

        task = PipelineTask(pipeline)
        await task.queue_frame(quiet_frame)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            transport.capture_participant_video(participant["id"], framerate=0)
            ir.set_participant_id(participant["id"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
