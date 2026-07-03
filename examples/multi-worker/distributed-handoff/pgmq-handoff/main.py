#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Main transport worker — run on Machine A.

Handles audio I/O (STT, TTS) and bridges frames to the bus. The LLM
workers run as separate processes (possibly on different
machines) connected via PGMQ on a shared Postgres database
(e.g. Supabase).

Usage::

    python main.py --database-url postgresql://...

Requirements:

- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DATABASE_URL (or ``--database-url``)
"""

import argparse
import os
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from loguru import logger
from pgmq.async_queue import PGMQueue

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus import BusBridgeProcessor
from pipecat.bus.network.pgmq import PgmqBus
from pipecat.evals.transport import EvalTransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.registry.types import WorkerReadyData
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.llm import LLMWorkerActivationArgs
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

MAIN_NAME = "acme"

transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def pgmq_from_url(database_url: str, *, pool_size: int = 4) -> PGMQueue:
    """Build a `PGMQueue` from a Postgres DSN string."""
    parsed = urlparse(database_url)
    if parsed.scheme not in ("postgres", "postgresql"):
        raise ValueError(f"Unsupported scheme '{parsed.scheme}' for database URL")
    return PGMQueue(
        host=parsed.hostname or "localhost",
        port=str(parsed.port or 5432),
        database=(parsed.path or "/postgres").lstrip("/") or "postgres",
        username=unquote(parsed.username or "postgres"),
        password=unquote(parsed.password or ""),
        pool_size=pool_size,
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    pgmq = pgmq_from_url(runner_args.cli_args.database_url)
    await pgmq.init()
    bus = PgmqBus(pgmq=pgmq, channel=runner_args.cli_args.channel)
    runner = WorkerRunner(bus=bus, handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    bridge = BusBridgeProcessor(
        bus=runner.bus,
        worker_name=MAIN_NAME,
        name=f"{MAIN_NAME}::BusBridge",
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            bridge,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # The remote LLM workers may take a moment to register on the bus.
    # We only activate ``greeter`` once *both* the client is connected
    # and the worker has been observed via the registry.
    state = {"client_connected": False, "greeter_ready": False}

    async def maybe_activate():
        if not (state["client_connected"] and state["greeter_ready"]):
            return
        await worker.activate_worker(
            "greeter",
            args=LLMWorkerActivationArgs(
                messages=[
                    {
                        "role": "developer",
                        "content": (
                            "Welcome the user to Acme Corp, mention the available "
                            "products and ask how you can help."
                        ),
                    },
                ],
            ),
        )

    async def on_greeter_ready(_data: WorkerReadyData) -> None:
        state["greeter_ready"] = True
        await maybe_activate()

    await runner.registry.watch("greeter", on_greeter_ready)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        state["client_connected"] = True
        await maybe_activate()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    parser = argparse.ArgumentParser(description="Main transport worker (PGMQ bus)")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL DSN (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--channel",
        default=os.getenv("PGMQ_CHANNEL", "pipecat_acme"),
        help="PGMQ channel prefix",
    )

    main(parser)
