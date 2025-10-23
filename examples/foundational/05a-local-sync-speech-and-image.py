#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import tkinter as tk

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    OutputAudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    URLImageRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.sync_parallel_pipeline import SyncParallelPipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaHttpTTSService
from pipecat.services.fal.image import FalImageGenService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.tk import TkLocalTransport, TkTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        tk_root = tk.Tk()
        tk_root.title("Calendar")

        runner = PipelineRunner()

        async def get_month_data(month):
            messages = [
                {
                    "role": "system",
                    "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit the description to one sentence, please.",
                }
            ]

            class ImageDescription(FrameProcessor):
                def __init__(self):
                    super().__init__()
                    self.text = ""

                async def process_frame(self, frame: Frame, direction: FrameDirection):
                    await super().process_frame(frame, direction)

                    if isinstance(frame, TextFrame):
                        self.text = frame.text
                    await self.push_frame(frame, direction)

            class AudioGrabber(FrameProcessor):
                def __init__(self):
                    super().__init__()
                    self.audio = bytearray()
                    self.frame = None

                async def process_frame(self, frame: Frame, direction: FrameDirection):
                    await super().process_frame(frame, direction)

                    if isinstance(frame, TTSAudioRawFrame):
                        self.audio.extend(frame.audio)
                        self.frame = OutputAudioRawFrame(
                            bytes(self.audio), frame.sample_rate, frame.num_channels
                        )
                    await self.push_frame(frame, direction)

            class ImageGrabber(FrameProcessor):
                def __init__(self):
                    super().__init__()
                    self.frame = None

                async def process_frame(self, frame: Frame, direction: FrameDirection):
                    await super().process_frame(frame, direction)

                    if isinstance(frame, URLImageRawFrame):
                        self.frame = frame
                    await self.push_frame(frame, direction)

            llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

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

            description = ImageDescription()

            audio_grabber = AudioGrabber()

            image_grabber = ImageGrabber()

            # With `SyncParallelPipeline` we synchronize audio and images by
            # pushing them basically in order (e.g. I1 A1 A1 A1 I2 A2 A2 A2 A2
            # I3 A3). To do that, each pipeline runs concurrently and
            # `SyncParallelPipeline` will wait for the input frame to be
            # processed.
            #
            # Note that `SyncParallelPipeline` requires the last processor in
            # each of the pipelines to be synchronous. In this case, we use
            # `CartesiaHttpTTSService` and `FalImageGenService` which make HTTP
            # requests and wait for the response.
            pipeline = Pipeline(
                [
                    llm,  # LLM
                    sentence_aggregator,  # Aggregates LLM output into full sentences
                    description,  # Store sentence
                    SyncParallelPipeline(
                        [tts, audio_grabber],  # Generate and store audio for the given sentence
                        [imagegen, image_grabber],  # Generate and storeimage for the given sentence
                    ),
                ]
            )

            task = PipelineTask(pipeline)
            await task.queue_frame(LLMContextFrame(LLMContext(messages)))
            await task.stop_when_done()

            await runner.run(task)

            return {
                "month": month,
                "text": description.text,
                "image": image_grabber.frame,
                "audio": audio_grabber.frame,
            }

        transport = TkLocalTransport(
            tk_root,
            TkTransportParams(
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_width=1024,
                video_out_height=1024,
            ),
        )

        pipeline = Pipeline([transport.output()])

        task = PipelineTask(pipeline)

        # We only specify a few months as we create tasks all at once and we
        # might get rate limited otherwise.
        months: list[str] = [
            "January",
            "February",
        ]

        # We create one task per month. This will be executed concurrently.
        month_tasks = [asyncio.create_task(get_month_data(month)) for month in months]

        # Now we wait for each month task in the order they're completed. The
        # benefit is we'll have as little delay as possible before the first
        # month, and likely no delay between months, but the months won't
        # display in order.
        async def show_images(month_tasks):
            for month_data_task in asyncio.as_completed(month_tasks):
                data = await month_data_task
                await task.queue_frames([data["image"], data["audio"]])

            await runner.stop_when_done()

        async def run_tk():
            while not task.has_finished():
                tk_root.update()
                tk_root.update_idletasks()
                await asyncio.sleep(0.1)

        await asyncio.gather(runner.run(task), show_images(month_tasks), run_tk())


if __name__ == "__main__":
    asyncio.run(main())
