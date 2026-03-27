#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Groq STT + Groq LLM + Lokutor TTS voice bot example.

This file follows the format used by other examples in `examples/foundational`.
It provides a runnable sample that uses:

- `GroqSTTService` for voice transcription from microphone input.
- `GroqLLMService` as the chat model.
- `LokutorTTSService` for speech output.

Run with:

    # install core dependencies
    uv sync --group dev --all-extras --no-extra gstreamer --no-extra krisp --no-extra local
    uv sync --extra lokutor

    # setup environment
    cp env.example .env
    export GROQ_API_KEY="your_groq_api_key"
    export LOKUTOR_API_KEY="your_lokutor_api_key"

    # run
    python examples/foundational/55zl-groq-stt-groq-llm-lokutor-tts.py -t webrtc

Then open `http://localhost:7860/client/`.

You can also run with other transport types:
- `-t daily` for Daily WebRTC
- `-t twilio` for telephony

The example is intentionally minimal and uses a conversational pipeline.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.lokutor.tts import LokutorTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Groq STT + Groq LLM + Lokutor TTS bot")

    stt = GroqSTTService(api_key=os.getenv("GROQ_API_KEY"))

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Speak naturally and avoid verbose lists."
            )
        ),
    )

    tts = LokutorTTSService(
        api_key=os.getenv("LOKUTOR_API_KEY"),
        voice_id="M1",
        params=LokutorTTSService.InputParams(
            language=Language.EN,
            speed=1.0,
            steps=5,
            visemes=False,
        ),
        sample_rate=44100,
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": "Introduce yourself and ask the user a question to start a short conversation.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
