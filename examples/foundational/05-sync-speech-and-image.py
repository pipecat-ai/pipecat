#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from dataclasses import dataclass

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.sync_parallel_pipeline import SyncParallelPipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaHttpTTSService
from pipecat.services.fal.image import FalImageGenService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


@dataclass
class MonthFrame(DataFrame):
    month: str

    def __str__(self):
        return f"{self.name}(month: {self.month})"


class MonthPrepender(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.most_recent_month = "Placeholder, month frame not yet received"
        self.prepend_to_next_text_frame = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MonthFrame):
            self.most_recent_month = frame.month
        elif self.prepend_to_next_text_frame and isinstance(frame, TextFrame):
            await self.push_frame(TextFrame(f"{self.most_recent_month}: {frame.text}"))
            self.prepend_to_next_text_frame = False
        elif isinstance(frame, LLMFullResponseStartFrame):
            self.prepend_to_next_text_frame = True
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Month Narration Bot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1024,
            ),
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        tts = CartesiaHttpTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        imagegen = FalImageGenService(
            params=FalImageGenService.InputParams(image_size="square_hd"),
            aiohttp_session=session,
            key=os.getenv("FAL_KEY"),
        )

        sentence_aggregator = SentenceAggregator()
        month_prepender = MonthPrepender()

        # With `SyncParallelPipeline` we synchronize audio and images by pushing
        # them basically in order (e.g. I1 A1 A1 A1 I2 A2 A2 A2 A2 I3 A3). To do
        # that, each pipeline runs concurrently and `SyncParallelPipeline` will
        # wait for the input frame to be processed.
        #
        # Note that `SyncParallelPipeline` requires the last processor in each
        # of the pipelines to be synchronous. In this case, we use
        # `CartesiaHttpTTSService` and `FalImageGenService` which make HTTP
        # requests and wait for the response.
        pipeline = Pipeline(
            [
                llm,  # LLM
                sentence_aggregator,  # Aggregates LLM output into full sentences
                SyncParallelPipeline(  # Run pipelines in parallel aggregating the result
                    [month_prepender, tts],  # Create "Month: sentence" and output audio
                    [imagegen],  # Generate image
                ),
                transport.output(),  # Transport output
            ]
        )

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
            frames.append(MonthFrame(month=month))
            frames.append(LLMMessagesFrame(messages))

        runner = PipelineRunner()

        task = PipelineTask(pipeline)

        await task.queue_frames(frames)

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
