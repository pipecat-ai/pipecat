#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MoQ voice-agent server: one process, many sessions, discovered by announcement.

Unlike ``transports-moq.py`` (one bot, browser kicked off via ``/start``), this
is a long-lived server that dials a relay once and spawns a fresh STT->LLM->TTS
pipeline for every client that announces itself -- the ``moq-boy`` pattern, no
control plane. See :mod:`pipecat.transports.moq.agent`.

Requirements:
    uv sync --extra moq --extra silero --extra deepgram --extra cartesia \
        --extra openai

Usage:
    # Local dev: run a moq relay (e.g. `just relay` in the moq repo on :4443),
    # then point the agent at it. Clients announce under anon/voice/client/*.
    uv run python examples/transports/transports-moq-agent.py \\
        --relay-url http://localhost:4443 --no-verify-ssl

    # Production: dial moq.dev and publish replies under an authenticated prefix.
    uv run python examples/transports/transports-moq-agent.py \\
        --relay-url https://relay.moq.dev --bot-prefix demo/voice/bot
"""

import argparse
import asyncio
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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.moq.agent import (
    DEFAULT_BOT_PREFIX,
    DEFAULT_CLIENT_PREFIX,
    DEFAULT_RELAY_URL,
    MOQAgentServer,
    MOQAgentSession,
)
from pipecat.transports.moq.transport import MOQParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


async def run_session_bot(transport: MOQAgentSession, client_id: str):
    """Build and run one client's voice pipeline until they disconnect."""
    logger.info(f"Starting bot for client {client_id!r}")

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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport):
        logger.info(f"Client {client_id!r} subscribed — starting conversation")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_disconnected")
    async def on_disconnected(transport):
        logger.info(f"Client {client_id!r} disconnected")
        await worker.cancel()

    @transport.event_handler("on_error")
    async def on_error(transport, message, exception):
        logger.error(f"MOQ error for {client_id!r}: {message}")

    # The server owns SIGINT; each session just runs its worker until the
    # client's mic track ends (which fires on_disconnected -> worker.cancel()).
    runner = WorkerRunner(handle_sigint=False)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        await transport.disconnect()


async def main():
    parser = argparse.ArgumentParser(description="MoQ voice-agent server")
    parser.add_argument("--relay-url", default=os.getenv("MOQ_RELAY_URL", DEFAULT_RELAY_URL))
    parser.add_argument("--client-prefix", default=DEFAULT_CLIENT_PREFIX)
    parser.add_argument("--bot-prefix", default=DEFAULT_BOT_PREFIX)
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Skip TLS verification (for self-signed local relays).",
    )
    args = parser.parse_args()

    params = MOQParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )

    server = MOQAgentServer(
        params,
        run_session_bot,
        relay_url=args.relay_url,
        client_prefix=args.client_prefix,
        bot_prefix=args.bot_prefix,
        verify_ssl=not args.no_verify_ssl,
        max_sessions=args.max_sessions,
    )

    logger.info("MoQ voice-agent server ready; waiting for clients to announce")
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
