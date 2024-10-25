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

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, OutputImageRawFrame, SystemFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaHttpTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyTransport

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

        if not isinstance(frame, SystemFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(
                OutputImageRawFrame(
                    image=self._speaking_image_bytes,
                    size=(1024, 1024),
                    format=self._speaking_image_format,
                )
            )
            await self.push_frame(frame)
            await self.push_frame(
                OutputImageRawFrame(
                    image=self._waiting_image_bytes,
                    size=(1024, 1024),
                    format=self._waiting_image_format,
                )
            )
        else:
            await self.push_frame(frame)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1024,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaHttpTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        image_sync_aggregator = ImageSyncAggregator(
            os.path.join(os.path.dirname(__file__), "assets", "speaking.png"),
            os.path.join(os.path.dirname(__file__), "assets", "waiting.png"),
        )

        pipeline = Pipeline(
            [
                transport.input(),
                image_sync_aggregator,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            participant_name = participant.get("info", {}).get("userName", "")
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([TextFrame(f"Hi there {participant_name}!")])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
