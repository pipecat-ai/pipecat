#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
from dataclasses import dataclass

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.sync_parallel_pipeline import SyncParallelPipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaHttpTTSService
from pipecat.services.fal.image import FalImageGenService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

load_dotenv(override=True)


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


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_width=1024,
        video_out_height=1024,
    ),
    "webrtc": lambda: TransportParams(
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_width=1024,
        video_out_height=1024,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the Calendar Month Narration bot using WebRTC transport.

    Args:
        webrtc_connection: The WebRTC connection to use
        room_name: Optional room name for display purposes
    """
    logger.info(f"Starting bot")

    # Create an HTTP session for API calls
    async with aiohttp.ClientSession() as session:
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
            frames.append(LLMContextFrame(LLMContext(messages)))

        task = PipelineTask(
            pipeline,
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        # Set up transport event handlers
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Start the month narration once connected
            await task.queue_frames(frames)

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        # Run the pipeline
        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
