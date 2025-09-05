#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMMessagesFrame, LLMMessagesUpdateFrame, LLMTextFrame
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver, FrameEndpoint
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)

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


class LLMRaceProcessor(FrameProcessor):
    """Manages racing between two LLMs - only allows frames from the first LLM to respond."""

    def __init__(self):
        super().__init__()
        self._current_llm_name = None

    def set_llm_name(self, name: str):
        """Set the name of the LLM this processor instance is handling."""
        self._current_llm_name = name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMTextFrame):
            if not LLMRaceProcessor._response_started:
                # First response wins the race
                LLMRaceProcessor._winning_llm_name = self._current_llm_name
                LLMRaceProcessor._response_started = True
                logger.info(
                    f"🏆 [LLM_RACE] {self._current_llm_name} wins the race! Text: '{frame.text}'"
                )
                await self.push_frame(frame, direction)
            elif LLMRaceProcessor._winning_llm_name == self._current_llm_name:
                # Continue allowing frames from winning LLM
                logger.info(f"✅ [LLM_RACE] {self._current_llm_name} continuing: '{frame.text}'")
                await self.push_frame(frame, direction)
            else:
                # Drop frames from losing LLM
                logger.info(
                    f"❌ [LLM_RACE] Dropping '{frame.text}' from losing LLM: {self._current_llm_name}"
                )
        else:
            # Always pass through non-LLM frames
            await self.push_frame(frame, direction)


# Class variables to share state between instances
LLMRaceProcessor._winning_llm_name = None
LLMRaceProcessor._response_started = False


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot with parallel LLM racing")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # Create two LLM instances for racing
    llm1 = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm2 = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    # Create shared context for both LLMs
    context = OpenAILLMContext(messages)
    context_aggregator = llm1.create_context_aggregator(context)

    # Make sure both LLMs share the same context - they should both process context frames
    # In a ParallelPipeline, the context frames will be duplicated to both branches

    # Create separate race processors for each LLM to track which one responds first
    race_processor1 = LLMRaceProcessor()
    race_processor1.set_llm_name("LLM1")

    race_processor2 = LLMRaceProcessor()
    race_processor2.set_llm_name("LLM2")

    # Create parallel LLM branches using ParallelPipeline
    parallel_llms = ParallelPipeline(
        [llm1, race_processor1],  # Branch 1: LLM1 -> race processor 1
        [llm2, race_processor2],  # Branch 2: LLM2 -> race processor 2
    )

    # Set up debug observers with filtering - only log LLM frames going to TTS
    debug_observer = DebugLogObserver(
        frame_types={LLMTextFrame: (CartesiaTTSService, FrameEndpoint.DESTINATION)}
    )
    llm_observer = LLMLogObserver()

    # Simple pipeline with parallel LLM processing
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech to text
            context_aggregator.user(),  # User responses (creates context frames for LLMs)
            parallel_llms,  # Parallel LLM processing
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
        # observers=[debug_observer, llm_observer],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Use a simpler approach - add message to context and push a context frame
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        # Create a new context with the updated messages
        updated_context = OpenAILLMContext(messages)
        await task.queue_frames([OpenAILLMContextFrame(context=updated_context)])

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
