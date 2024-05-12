#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

import daily

from pipecat.frames.frames import (
    AppFrame,
    Frame,
    ImageRawFrame,
    TextFrame,
    EndFrame,
    LLMMessagesFrame,
    LLMResponseStartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.gated import GatedAggregator
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.aggregators.parallel_task import ParallelTask
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.fal import FalImageGenService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class MonthFrame(AppFrame):
    def __init__(self, month):
        super().__init__()
        self.metadata["month"] = month

    @ property
    def month(self) -> str:
        return self.metadata["month"]

    def __str__(self):
        return f"{self.name}(month: {self.month})"

    month: str


class MonthPrepender(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.most_recent_month = "Placeholder, month frame not yet received"
        self.prepend_to_next_text_frame = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, MonthFrame):
            self.most_recent_month = frame.month
        elif self.prepend_to_next_text_frame and isinstance(frame, TextFrame):
            await self.push_frame(TextFrame(f"{self.most_recent_month}: {frame.data}"))
            self.prepend_to_next_text_frame = False
        elif isinstance(frame, LLMResponseStartFrame):
            self.prepend_to_next_text_frame = True
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            None,
            "Month Narration Bot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1024
            )
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview")

        imagegen = FalImageGenService(
            params=FalImageGenService.InputParams(
                image_size="square_hd"
            ),
            aiohttp_session=session,
            key=os.getenv("FAL_KEY"),
        )

        gated_aggregator = GatedAggregator(
            gate_open_fn=lambda frame: isinstance(frame, ImageRawFrame),
            gate_close_fn=lambda frame: isinstance(frame, LLMResponseStartFrame),
            start_open=False
        )

        sentence_aggregator = SentenceAggregator()
        month_prepender = MonthPrepender()
        llm_full_response_aggregator = LLMFullResponseAggregator()

        pipeline = Pipeline([
            llm,
            sentence_aggregator,
            ParallelTask(
                [month_prepender, tts],
                [llm_full_response_aggregator, imagegen]
            ),
            gated_aggregator,
            transport.output()
        ])

        frames = []
        for month in [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]:
            messages = [
                {
                    "role": "system",
                    "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit the description to one sentence, please.",
                }
            ]
            frames.append(MonthFrame(month))
            frames.append(LLMMessagesFrame(messages))

        frames.append(EndFrame())

        runner = PipelineRunner()

        task = PipelineTask(pipeline)

        await task.queue_frames(frames)

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
