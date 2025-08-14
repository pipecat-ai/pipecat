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
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)

# Create VAD parameters optimized for quiet speakers


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


class TranscriptionLogger(FrameProcessor):
    """Custom processor that logs transcription frames."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Only log TranscriptionFrame objects
        if isinstance(frame, TranscriptionFrame):
            logger.info(f"[TRANSCRIPTION]: {frame.text}")

        # Always pass the frame through to maintain pipeline flow
        await self.push_frame(frame, direction)


class InterventionProcessor(FrameProcessor):
    """Custom processor that logs LLM response frames."""

    def __init__(self):
        super().__init__()
        self._timer_task = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Log LLM response start frames
        if isinstance(frame, LLMFullResponseStartFrame):
            logger.info(f"[LLM_START]: Starting LLM response")

            # Cancel any existing timer
            if self._timer_task and not self._timer_task.done():
                self._timer_task.cancel()

            # Start a new 500ms timer
            self._timer_task = asyncio.create_task(self._log_after_delay())

        # Cancel timer if bot started speaking before 500ms
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info(f"[BOT_SPEAKING]: Bot started speaking, canceling intervention timer")
            if self._timer_task and not self._timer_task.done():
                self._timer_task.cancel()

        # Log LLM text frames
        elif isinstance(frame, LLMTextFrame):
            logger.info(f"[LLM_TEXT]: {frame.text}")

        # Always pass the frame through to maintain pipeline flow
        await self.push_frame(frame, direction)

    async def _log_after_delay(self):
        """Log a message after 500ms delay."""
        try:
            await asyncio.sleep(0.5)  # 500ms
            logger.info(f"500ms passed since LLMFullResponseStartFrame")
            await self.queue_frame(TTSSpeakFrame("um..."))
        except asyncio.CancelledError:
            # Timer was cancelled, which is fine
            pass


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
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

    # Create transcription logger instance
    transcription_logger = TranscriptionLogger()

    # Create LLM logger instance
    intervention = InterventionProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            transcription_logger,  # Log transcription frames
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            intervention,  # Log LLM response frames
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
