#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from PIL import Image

from pipecat.frames.frames import ImageRawFrame, Frame, SystemFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from pipecat.transports.services.daily import DailyParams
from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class ImageSyncAggregator(FrameProcessor):
    def __init__(self, speaking_path: str, waiting_path: str):
        super().__init__()
        self._speaking_image = Image.open(speaking_path)
        self._speaking_image_format = self._speaking_image.format
        self._speaking_image_bytes = self._speaking_image.tobytes()

        self._waiting_image = Image.open(waiting_path)
        self._waiting_image_format = self._waiting_image.format
        self._waiting_image_bytes = self._waiting_image.tobytes()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, SystemFrame):
            await self.push_frame(ImageRawFrame(image=self._speaking_image_bytes, size=(1024, 1024), format=self._speaking_image_format))
            await self.push_frame(frame)
            await self.push_frame(ImageRawFrame(image=self._waiting_image_bytes, size=(1024, 1024), format=self._waiting_image_format))
        else:
            await self.push_frame(frame)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1024,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        image_sync_aggregator = ImageSyncAggregator(
            os.path.join(os.path.dirname(__file__), "assets", "speaking.png"),
            os.path.join(os.path.dirname(__file__), "assets", "waiting.png"),
        )

        pipeline = Pipeline([
            transport.input(),
            image_sync_aggregator,
            tma_in,
            llm,
            tts,
            transport.output(),
            tma_out
        ])

        task = PipelineTask(pipeline)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            participant_name = participant["info"]["userName"] or ''
            transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([TextFrame(f"Hi, this is {participant_name}.")])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
