#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, TextFrame, UserImageRequestFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.user_response import UserResponseAggregator
from pipecat.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.google import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class UserImageRequester(FrameProcessor):
    def __init__(self, participant_id: str | None = None):
        super().__init__()
        self._participant_id = participant_id

    def set_participant_id(self, participant_id: str):
        self._participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._participant_id and isinstance(frame, TextFrame):
            await self.push_frame(
                UserImageRequestFrame(self._participant_id), FrameDirection.UPSTREAM
            )
        await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Describe participant video",
            DailyParams(
                audio_in_enabled=True,  # This is so Silero VAD can get audio data
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        user_response = UserResponseAggregator()

        image_requester = UserImageRequester()

        vision_aggregator = VisionImageFrameAggregator()

        google = GoogleLLMService(
            model="gemini-1.5-flash-latest", api_key=os.getenv("GOOGLE_API_KEY")
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await tts.say("Hi there! Feel free to ask me what I see.")
            await transport.capture_participant_video(participant["id"], framerate=0)
            await transport.capture_participant_transcription(participant["id"])
            image_requester.set_participant_id(participant["id"])

        pipeline = Pipeline(
            [
                transport.input(),
                user_response,
                image_requester,
                vision_aggregator,
                google,
                tts,
                transport.output(),
            ]
        )

        task = PipelineTask(pipeline)

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
