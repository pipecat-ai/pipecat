#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Haru, a full-duplex speech-to-speech model, run in this process.

Haru listens and speaks at the same time and decides on its own when to talk,
so the pipeline has no VAD, no STT and no TTS. It is a pretrained base model
for natural conversation: it plays a character, it is not an assistant.

The model runs on this machine: with MLX on Apple silicon
(``uv add "pipecat-ai[haru-mlx]"``) or with PyTorch on an NVIDIA GPU with
about 12 GB of memory (``uv add "pipecat-ai[haru-torch]"``). The checkpoint,
about 10 GB, is downloaded from Hugging Face on first use.

Set ``HARU_VOICE_REFERENCE`` to a clip of the voice the model should speak
with: one speaker, 3 seconds to 3 minutes of clean speech.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.haru.local import HaruLocalLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# Haru reads its character from a card. It follows a persona and a situation,
# not instructions.
CHARACTER_CARD = """Haru is a 29-year-old nurse in Denver. She is warm, chatty and laughs \
easily. She is on a phone call with an old friend from high school, catching up after a \
long time."""

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "livekit": lambda: LiveKitParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    # The model starts loading here, before a client connects.
    llm = HaruLocalLLMService(
        voice_reference=os.getenv("HARU_VOICE_REFERENCE"),
        settings=HaruLocalLLMService.Settings(
            system_instruction=CHARACTER_CARD,
            speak_first=True,
        ),
    )

    # Haru is full-duplex: it handles turn-taking and being interrupted
    # itself, so there is no local VAD and interruptions are never broadcast.
    # It produces no transcript of the user, so the context records only what
    # the model says.
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            user_aggregator,
            llm,  # LLM
            transport.output(),  # Transport bot output
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            # Haru works at 24 kHz. The default input rate is 16 kHz, which
            # would hand the model upsampled audio with nothing above 8 kHz.
            audio_in_sample_rate=24000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Start the conversation.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    @llm.event_handler("on_session_started")
    async def on_session_started(llm):
        logger.info("Haru is listening")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"Transcript: assistant: {message.content}")

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
