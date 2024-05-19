#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import random
import sys

from PIL import Image

from pipecat.frames.frames import (
    Frame,
    SystemFrame,
    TextFrame,
    ImageRawFrame,
    SpriteFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import (
    LLMUserContextAggregator,
    LLMAssistantContextAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


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
        sprites[file] = ImageRawFrame(image=img.tobytes(), size=img.size, format=img.format)

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = sprites["sc-listen-1.png"]

# When the bot is talking, build an animation from two sprites
talking_list = [sprites["sc-default.png"], sprites["sc-talk.png"]]
talking = [random.choice(talking_list) for x in range(30)]
talking_frame = SpriteFrame(talking)

# TODO: Support "thinking" as soon as we get a valid transcript, while LLM
# is processing
thinking_list = [
    sprites["sc-think-1.png"],
    sprites["sc-think-2.png"],
    sprites["sc-think-3.png"],
    sprites["sc-think-4.png"],
]
thinking_frame = SpriteFrame(thinking_list)


class NameCheckFilter(FrameProcessor):
    def __init__(self, names: list[str]):
        super().__init__()
        self._names = names
        self._sentence = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        content: str = ""

        # TODO: split up transcription by participant
        if isinstance(frame, TranscriptionFrame):
            content = frame.text
            self._sentence += content
            if self._sentence.endswith((".", "?", "!")):
                if any(name in self._sentence for name in self._names):
                    await self.push_frame(TextFrame(self._sentence))
                    self._sentence = ""
                else:
                    self._sentence = ""
        else:
            await self.push_frame(frame, direction)


class ImageSyncAggregator(FrameProcessor):

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(talking_frame)
        await self.push_frame(frame)
        await self.push_frame(quiet_frame)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Santa Cat",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=720,
                camera_out_height=1280,
                camera_out_framerate=10,
                transcription_enabled=True
            )
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview")

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="jBpfuIE2acCO8z3wKNLl",
        )
        isa = ImageSyncAggregator()

        messages = [
            {
                "role": "system",
                "content": "You are Santa Cat, a cat that lives in Santa's workshop at the North Pole. You should be clever, and a bit sarcastic. You should also tell jokes every once in a while.  Your responses should only be a few sentences long.",
            },
        ]

        tma_in = LLMUserContextAggregator(messages)
        tma_out = LLMAssistantContextAggregator(messages)
        ncf = NameCheckFilter(["Santa Cat", "Santa"])

        pipeline = Pipeline([
            transport.input(),
            isa,
            ncf,
            tma_in,
            llm,
            tts,
            transport.output(),
            tma_out
        ])

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Send some greeting at the beginning.
            await tts.say("Hi! If you want to talk to me, just say 'hey Santa Cat'.")
            transport.capture_participant_transcription(participant["id"])

        async def starting_image():
            await transport.send_image(quiet_frame)

        runner = PipelineRunner()

        task = PipelineTask(pipeline)

        await asyncio.gather(runner.run(task), starting_image())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
