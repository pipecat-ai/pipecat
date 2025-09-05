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
from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
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
    """Processor that sends frames to two LLMs in parallel and uses the first response."""

    def __init__(self, llm1: OpenAILLMService, llm2: OpenAILLMService):
        super().__init__()
        self._llm1 = llm1
        self._llm2 = llm2
        self._race_counter = 0
        self._active_races = {}  # race_id -> winner_name

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Check if this is a frame we should race through both LLMs
        if isinstance(frame, LLMMessagesFrame):
            race_id = self._race_counter
            self._race_counter += 1

            logger.info(f"[LLM_RACE {race_id}] Starting parallel processing")

            # Create a result collector for this race
            race_result = asyncio.Event()
            winning_frames = []

            async def llm_runner(llm: OpenAILLMService, name: str):
                """Run LLM and collect results."""
                try:
                    # Create a frame collector
                    class FrameCollector(FrameProcessor):
                        def __init__(self):
                            super().__init__()
                            self.collected_frames = []

                        async def process_frame(self, frame, direction):
                            self.collected_frames.append(frame)

                    collector = FrameCollector()

                    # Temporarily link LLM to collector
                    llm.link(collector)

                    # Process the frame
                    await llm.process_frame(frame, FrameDirection.DOWNSTREAM)

                    # Check if we won the race
                    if race_id not in self._active_races:
                        self._active_races[race_id] = name
                        winning_frames.extend(collector.collected_frames)
                        race_result.set()
                        logger.info(
                            f"[LLM_RACE {race_id}] {name} WON with {len(collector.collected_frames)} frames!"
                        )
                    else:
                        logger.info(f"[LLM_RACE {race_id}] {name} lost")

                except Exception as e:
                    logger.error(f"[LLM_RACE {race_id}] Error in {name}: {e}")

            # Start both LLMs racing
            task1 = asyncio.create_task(llm_runner(self._llm1, "LLM1"))
            task2 = asyncio.create_task(llm_runner(self._llm2, "LLM2"))

            # Wait for the first one to complete
            await race_result.wait()

            # Cancel the slower task
            task1.cancel()
            task2.cancel()

            # Push the winning frames
            for winning_frame in winning_frames:
                await self.push_frame(winning_frame, direction)

        else:
            # Pass through non-LLM frames
            await self.push_frame(frame, direction)


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

    # Create a second context aggregator that shares the same context
    context_aggregator2 = llm2.create_context_aggregator(context)

    # Create race processor with both LLMs
    race_processor = LLMRaceProcessor(llm1, llm2)

    # Simple pipeline - the race processor handles the parallel LLM execution internally
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech to text
            context_aggregator.user(),  # User responses (creates context frames for LLMs)
            race_processor,  # Parallel LLM racing processor
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
        await task.queue_frames([context_aggregator.user().get_context_frame()])

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
