#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMRunFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


class DelayProcessor(FrameProcessor):
    """Custom processor that queues LLM text frames until response is complete.

    This creates a more natural conversation flow by preventing the agent from
    responding immediately after the user stops speaking. It queues all LLMTextFrames
    until it sees an LLMFullResponseEndFrame, then waits for the specified delay
    before releasing all queued frames at once.
    """

    def __init__(self, *, delay_seconds: float = 1.0, **kwargs) -> None:
        """Initialize the DelayProcessor.

        Args:
            delay_seconds: Number of seconds to delay before releasing queued frames (default: 1.0)
        """
        super().__init__(**kwargs)
        self._delay_seconds = delay_seconds
        self._queued_frames = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames, queuing LLM text frames until response is complete.

        Args:
            frame: The frame to process
            direction: Direction of the frame in the pipeline
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            # Queue LLM text frames instead of pushing them immediately
            logger.debug(f"Queuing LLMTextFrame: {frame.text}")
            self._queued_frames.append((frame, direction))
        elif isinstance(frame, LLMFullResponseEndFrame):
            # When we see the end frame, wait for delay then push all queued frames
            logger.debug(
                f"LLM response complete, delaying {self._delay_seconds} seconds before releasing {len(self._queued_frames)} queued frames"
            )
            await asyncio.sleep(self._delay_seconds)

            # Push all queued LLM text frames
            for queued_frame, queued_direction in self._queued_frames:
                logger.debug(f"Releasing queued LLMTextFrame: {queued_frame.text}")
                await self.push_frame(queued_frame, queued_direction)

            # Clear the queue
            self._queued_frames.clear()

            # Push the end frame
            logger.debug("Pushing LLMFullResponseEndFrame")
            await self.push_frame(frame, direction)
        else:
            # Push all other frames immediately
            await self.push_frame(frame, direction)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Create delay processor to add 1-second delay before agent responses
    delay_processor = DelayProcessor(delay_seconds=1.0)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            delay_processor,  # Add delay before TTS
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
