#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MOQ (Media over QUIC) transport example.

This example demonstrates using the MOQ transport for real-time voice
conversations over QUIC. MOQ provides WebRTC-like latency without WebRTC
constraints, using QUIC for prioritization and partial reliability.

Requirements:
    uv sync --extra moq --extra silero --extra deepgram --extra cartesia \
        --extra openai --extra runner

Usage:
    # Local dev — bot is its own MOQ server, mints a self-signed cert
    # for `localhost`, browser pins the fingerprint via /api/config.
    # No separate relay needed:
    uv run python examples/transports/transports-moq.py \\
        -t moq --moq-serve --moq-tls-generate localhost

    # Connect to a remote relay (CA-signed cert, no pinning needed):
    uv run python examples/transports/transports-moq.py \\
        -t moq --moq-connect https://moq.example.com:4080/moq

    # With a custom namespace (different "room"):
    uv run python examples/transports/transports-moq.py \\
        -t moq --moq-serve --moq-tls-generate localhost --moq-namespace my-room

    # Then open http://localhost:7860 and click Connect.

    # Can also run with other transports:
    uv run python examples/transports/transports-moq.py -t webrtc
    uv run python examples/transports/transports-moq.py -t daily
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.moq.transport import MOQParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# Transport-specific parameters using lambdas for deferred creation
transport_params = {
    "moq": lambda: MOQParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the bot with the given transport."""
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model="gpt-4o",
            system_instruction=(
                "You are a helpful assistant in a real-time voice call. "
                "Your goal is to demonstrate your capabilities in a succinct way. "
                "Your output will be spoken aloud, so avoid special characters that can't easily "
                "be spoken, such as emojis or bullet points. Respond to what the user said in a "
                "creative and helpful way."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport):
        logger.info("Client subscribed — starting conversation")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_disconnected")
    async def on_disconnected(transport):
        logger.info("Disconnected from MOQ relay")
        await worker.cancel()

    @transport.event_handler("on_error")
    async def on_error(transport, message, exception):
        logger.error(f"MOQ error: {message}")

    # MOQInputTransport.start() auto-connects to the relay when the
    # pipeline starts, so we don't dial transport.connect() here.
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        await transport.disconnect()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
