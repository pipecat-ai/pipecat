#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.user_response import UserResponseAggregator
from pipecat.processors.aggregators.vision_image_frame import VisionImageFrameAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.azure import AzureTTSService, AzureLLMService
from pipecat.services.google import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from main.utils.prompts import agent_system_prompt

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
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            ),
        )

        user_response = UserResponseAggregator()

        image_requester = UserImageRequester()

        vision_aggregator = VisionImageFrameAggregator()

        google = GoogleLLMService(
            model="gemini-1.5-pro-002", api_key='AIzaSyCOq3Qny27BU5I9XC2Qsu73DtQ85vxXJ6E'
        )

        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region="eastus",
            voice="en-US-AvaMultilingualNeural", # en-US-AvaMultilingualNeural en-US-ShimmerMultilingualNeural
            
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await tts.say("Hi there! Feel free to ask me what I see.")
            transport.capture_participant_transcription(participant["id"])
            # transport.capture_participant_video(participant["id"], framerate=0, video_source="screenVideo")
            # transport.capture_participant_transcription(participant["id"])
            # image_requester.set_participant_id(participant["id"])

        @transport.event_handler("on_participant_updated")
        async def on_participant_updated(transport, participant):

            screen_video = participant.get("media", {}).get("screenVideo", {})
            subscribed = screen_video.get("subscribed")
            state = screen_video.get("state")
            if subscribed == "unsubscribed" and state == "receivable":
                transport.capture_participant_video(
                    participant["id"], framerate=0, video_source="screenVideo")
                image_requester.set_participant_id(participant["id"])


        messages = [
            {
                "role": "system",
                "content": agent_system_prompt
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = google.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                image_requester,
                vision_aggregator,
                google,
                tts,
                transport.output(),
                context_aggregator.assistant()
            ]
        )

        task = PipelineTask(pipeline)

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
