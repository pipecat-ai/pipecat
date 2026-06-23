#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deployable MoQ voice-agent server (the ``moq-voice-agent`` entry point).

A packaged, long-lived :class:`MOQAgentServer` that discovers clients by MoQ
announcement and runs an STT->LLM->TTS turn per client. This is the installable
counterpart to ``examples/transports/transports-moq-agent.py``: it ships as a
console script so it can be built (uv/uv2nix or an OCI image) and run as a
systemd service co-located on a relay, dialing the relay's internal Unix socket
(``--relay-url unix:///run/moq/internal.sock``).

The default pipeline is Deepgram (STT) + OpenAI (LLM) + Cartesia (TTS), all
configured from the environment so the same image serves every deployment:

    DEEPGRAM_API_KEY / OPENAI_API_KEY / CARTESIA_API_KEY  (required)
    MOQ_VOICE_LLM_MODEL      (default: gpt-4o)
    MOQ_VOICE_TTS_VOICE      (default: British Reading Lady)
    MOQ_VOICE_SYSTEM_PROMPT  (default: a short real-time-voice instruction)

Run ``moq-voice-agent --help`` for the transport flags.
"""

import argparse
import asyncio
import os

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

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant in a real-time voice call. "
    "Your goal is to demonstrate your capabilities in a succinct way. "
    "Your output will be spoken aloud, so avoid special characters that can't easily "
    "be spoken, such as emojis or bullet points. Respond to what the user said in a "
    "creative and helpful way."
)
# Cartesia "British Reading Lady".
DEFAULT_TTS_VOICE = "71a7ad14-091c-4e8e-a314-022ece01c121"


async def run_session_bot(transport: MOQAgentSession, client_id: str):
    """Build and run one client's voice pipeline until they disconnect."""
    logger.info(f"Starting bot for client {client_id!r}")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("MOQ_VOICE_TTS_VOICE", DEFAULT_TTS_VOICE),
        ),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model=os.getenv("MOQ_VOICE_LLM_MODEL", "gpt-4o"),
            system_instruction=os.getenv("MOQ_VOICE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport):
        logger.info(f"Client {client_id!r} subscribed -- starting conversation")
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

    runner = WorkerRunner(handle_sigint=False)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        await transport.disconnect()


async def _run(args: argparse.Namespace) -> None:
    params = MOQParams(audio_in_enabled=True, audio_out_enabled=True)
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


def main() -> None:
    """Console-script entry point (``moq-voice-agent``)."""
    parser = argparse.ArgumentParser(description="MoQ voice-agent server")
    parser.add_argument("--relay-url", default=os.getenv("MOQ_RELAY_URL", DEFAULT_RELAY_URL))
    parser.add_argument("--client-prefix", default=DEFAULT_CLIENT_PREFIX)
    parser.add_argument("--bot-prefix", default=DEFAULT_BOT_PREFIX)
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Skip TLS verification (self-signed relays; moot over a Unix socket).",
    )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
