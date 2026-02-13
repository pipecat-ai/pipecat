#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MOQ (Media over QUIC) transport example.

This example demonstrates using the MOQ transport for real-time voice
conversations over QUIC, connecting to a MOQ relay server. It uses the
unified runner pattern that works with Daily, WebRTC, and MOQ transports.

MOQ provides WebRTC-like latency without WebRTC constraints, using QUIC
for prioritization and partial reliability.

Requirements:
    uv sync --extra moq --extra silero --extra deepgram --extra cartesia \
        --extra openai --extra runner

    # You also need a MOQ relay running locally. Clone moq-relay from
    # https://github.com/kixelated/moq and then run scripts/moq-dev-setup.sh
    # from this repo, pointing at the relay checkout. The script generates
    # a self-signed cert, symlinks it into both repos, and prints the
    # exact relay + bot run commands to copy.
    #
    #   git clone https://github.com/kixelated/moq.git ../moq
    #   ./scripts/moq-dev-setup.sh ../moq
    #
    # Then in two terminals run the commands the script printed (relay
    # binds QUIC on UDP [::]:4080 with --auth-public '').

Usage:
    # Run with MOQ transport (connects to local relay set up by the script):
    uv run python examples/transports/transports-moq.py \\
        -t moq --moq-cert moq-cert.pem --moq-insecure --moq-path /

    # Connect to a remote relay (CA-signed cert, no pinning needed):
    uv run python examples/transports/transports-moq.py \\
        -t moq --moq-host moq.example.com

    # With a custom namespace (different "room"):
    uv run python examples/transports/transports-moq.py \\
        -t moq --moq-cert moq-cert.pem --moq-insecure --moq-namespace my-room

    # Then open the browser client at http://localhost:7860 and click Connect.

    # Can also run with other transports (no relay needed):
    uv run python examples/transports/transports-moq.py -t webrtc
    uv run python examples/transports/transports-moq.py -t daily
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
from pipecat.runner.types import MOQRunnerArguments, RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.moq import MOQParams
from pipecat.transports.moq.protocol import MOQRole

load_dotenv(override=True)

# Transport-specific parameters using lambdas for deferred creation
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "moq": lambda: MOQParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        role=MOQRole.PUBSUB,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the bot with the given transport."""
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant in a real-time voice call. "
            "Your goal is to demonstrate your capabilities in a succinct way. "
            "Your output will be spoken aloud, so avoid special characters that can't easily be "
            "spoken, such as emojis or bullet points. Respond to what the user said in a creative "
            "and helpful way.",
        },
    ]

    context = LLMContext(messages)
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # For MOQ, we need to handle connection and events differently
    if isinstance(runner_args, MOQRunnerArguments):

        @transport.event_handler("on_connected")
        async def on_connected(transport):
            logger.info("Connected to MOQ relay (waiting for client to join)")
            if runner_args.ready_event is not None:
                runner_args.ready_event.set()

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport):
            logger.info("Client subscribed — starting conversation")
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_disconnected")
        async def on_disconnected(transport):
            logger.info("Disconnected from MOQ relay")
            await task.cancel()

        @transport.event_handler("on_error")
        async def on_error(transport, message, exception):
            logger.error(f"MOQ error: {message}")

        # MOQInputTransport.start() auto-connects to the relay when the
        # pipeline starts, so we don't dial transport.connect() here.
        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        try:
            await runner.run(task)
        finally:
            await transport.disconnect()
    else:
        # Daily and WebRTC use on_client_connected/on_client_disconnected
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
