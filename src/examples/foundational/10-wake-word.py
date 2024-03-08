import aiohttp
import asyncio
import logging
import os
import random
from typing import AsyncGenerator
from PIL import Image

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.pipeline.aggregators import (
    LLMUserContextAggregator,
    LLMAssistantContextAggregator,
)
from dailyai.pipeline.frames import (
    Frame,
    TextFrame,
    ImageFrame,
    SpriteFrame,
    TranscriptionQueueFrame,
)
from dailyai.services.ai_services import AIService
from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

sprites = {}
image_files = [
    "sc-default.png",
    "sc-talk.png",
    "sc-listen-1.png",
    "sc-think-1.png",
    "sc-think-2.png",
    "sc-think-3.png",
    "sc-think-4.png",
]

script_dir = os.path.dirname(__file__)

for file in image_files:
    # Build the full path to the image file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites[file] = img.tobytes()

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = ImageFrame("", sprites["sc-listen-1.png"])
# When the bot is talking, build an animation from two sprites
talking_list = [sprites["sc-default.png"], sprites["sc-talk.png"]]
talking = [random.choice(talking_list) for x in range(30)]
talking_frame = SpriteFrame(images=talking)

# TODO: Support "thinking" as soon as we get a valid transcript, while LLM is processing
thinking_list = [
    sprites["sc-think-1.png"],
    sprites["sc-think-2.png"],
    sprites["sc-think-3.png"],
    sprites["sc-think-4.png"],
]
thinking_frame = SpriteFrame(images=thinking_list)


class TranscriptFilter(AIService):
    def __init__(self, bot_participant_id=None):
        self.bot_participant_id = bot_participant_id

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, TranscriptionQueueFrame):
            if frame.participantId != self.bot_participant_id:
                yield frame


class NameCheckFilter(AIService):
    def __init__(self, names: list[str]):
        self.names = names
        self.sentence = ""

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        content: str = ""

        # TODO: split up transcription by participant
        if isinstance(frame, TextFrame):
            content = frame.text

        self.sentence += content
        if self.sentence.endswith((".", "?", "!")):
            if any(name in self.sentence for name in self.names):
                out = self.sentence
                self.sentence = ""
                yield TextFrame(out)
            else:
                out = self.sentence
                self.sentence = ""


class ImageSyncAggregator(AIService):
    def __init__(self):
        pass

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        yield talking_frame
        yield frame
        yield quiet_frame


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
            room_url,
            token,
            "Santa Cat",
            duration_minutes=3,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=True,
            camera_width=720,
            camera_height=1280,
        )
        transport._mic_enabled = True
        transport._mic_sample_rate = 16000
        transport._camera_enabled = True
        transport._camera_width = 720
        transport._camera_height = 1280

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_CHATGPT_API_KEY"), model="gpt-4-turbo-preview"
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="jBpfuIE2acCO8z3wKNLl",
        )
        isa = ImageSyncAggregator()

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await tts.say(
                "Hi! If you want to talk to me, just say 'hey Santa Cat'.",
                transport.send_queue,
            )

        async def handle_transcriptions():
            messages = [
                {
                    "role": "system",
                    "content": "You are Santa Cat, a cat that lives in Santa's workshop at the North Pole. You should be clever, and a bit sarcastic. You should also tell jokes every once in a while.  Your responses should only be a few sentences long.",
                },
            ]

            tma_in = LLMUserContextAggregator(messages, transport._my_participant_id)
            tma_out = LLMAssistantContextAggregator(
                messages, transport._my_participant_id
            )
            tf = TranscriptFilter(transport._my_participant_id)
            ncf = NameCheckFilter(["Santa Cat", "Santa"])
            await tts.run_to_queue(
                transport.send_queue,
                isa.run(
                    tma_out.run(
                        llm.run(
                            tma_in.run(ncf.run(tf.run(transport.get_receive_frames())))
                        )
                    )
                ),
            )

        async def starting_image():
            await transport.send_queue.put(quiet_frame)

        transport.transcription_settings["extra"]["punctuate"] = True
        await asyncio.gather(transport.run(), handle_transcriptions(), starting_image())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
