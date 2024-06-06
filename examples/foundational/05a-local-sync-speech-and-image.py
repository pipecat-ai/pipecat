#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys

import tkinter as tk

from pipecat.frames.frames import AudioRawFrame, Frame, URLImageRawFrame, LLMMessagesFrame, TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.fal import FalImageGenService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.tk import TkLocalTransport

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        tk_root = tk.Tk()
        tk_root.title("Calendar")

        runner = PipelineRunner()

        async def get_month_data(month):
            messages = [{"role": "system", "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit the description to one sentence, please.", }]

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

                async def process_frame(self, frame: Frame, direction: FrameDirection):
                    await super().process_frame(frame, direction)

                    if isinstance(frame, AudioRawFrame):
                        self.audio.extend(frame.audio)
                        self.frame = AudioRawFrame(
                            bytes(self.audio), frame.sample_rate, frame.num_channels)

            class ImageGrabber(FrameProcessor):
                def __init__(self):
                    super().__init__()
                    self.frame = None

                async def process_frame(self, frame: Frame, direction: FrameDirection):
                    await super().process_frame(frame, direction)

                    if isinstance(frame, URLImageRawFrame):
                        self.frame = frame

            llm = OpenAILLMService(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o")

            tts = ElevenLabsTTSService(
                aiohttp_session=session,
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id=os.getenv("ELEVENLABS_VOICE_ID"))

            imagegen = FalImageGenService(
                params=FalImageGenService.InputParams(
                    image_size="square_hd"
                ),
                aiohttp_session=session,
                key=os.getenv("FAL_KEY"))

            aggregator = LLMFullResponseAggregator()

            description = ImageDescription()

            audio_grabber = AudioGrabber()

            image_grabber = ImageGrabber()

            pipeline = Pipeline([
                llm,
                aggregator,
                description,
                ParallelPipeline([tts, audio_grabber],
                                 [imagegen, image_grabber])
            ])

            task = PipelineTask(pipeline)
            await task.queue_frame(LLMMessagesFrame(messages))
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
            TransportParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1024))

        pipeline = Pipeline([transport.output()])

        task = PipelineTask(pipeline)

        # We only specify 5 months as we create tasks all at once and we might
        # get rate limited otherwise.
        months: list[str] = [
            "January",
            "February",
            # "March",
            # "April",
            # "May",
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
