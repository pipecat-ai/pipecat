#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys
import wave

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    LLMFullResponseEndFrame,
    LLMMessagesFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMUserResponseAggregator,
    LLMAssistantResponseAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.logger import FrameLogger
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


sounds = {}
sound_files = ["ding1.wav", "ding2.wav"]

script_dir = os.path.dirname(__file__)

for file in sound_files:
    # Build the full path to the image file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the image and convert it to bytes
    with wave.open(full_path) as audio_file:
        sounds[file] = AudioRawFrame(audio_file.readframes(-1),
                                     audio_file.getframerate(), audio_file.getnchannels())


class OutboundSoundEffectWrapper(FrameProcessor):

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(sounds["ding1.wav"])
            # In case anything else downstream needs it
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class InboundSoundEffectWrapper(FrameProcessor):

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMMessagesFrame):
            await self.push_frame(sounds["ding2.wav"])
            # In case anything else downstream needs it
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o")

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="ErXwobaYiN019PkySvjV",
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way.",
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)
        out_sound = OutboundSoundEffectWrapper()
        in_sound = InboundSoundEffectWrapper()
        fl = FrameLogger("LLM Out")
        fl2 = FrameLogger("Transcription In")

        pipeline = Pipeline([
            transport.input(),
            tma_in,
            in_sound,
            fl2,
            llm,
            fl,
            tts,
            out_sound,
            transport.output(),
            tma_out
        ])

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            await tts.say("Hi, I'm listening!")
            await transport.send_audio(sounds["ding1.wav"])

        runner = PipelineRunner()

        task = PipelineTask(pipeline)

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
