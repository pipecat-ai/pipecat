#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Main transport worker with a WebSocket proxy to a remote LLM server.

Handles audio I/O (STT, TTS) and bridges frames to the bus. A
`WebSocketProxyClient` forwards bus messages to a remote LLM
server (see ``assistant.py``) over WebSocket.

Usage::

    python main.py --remote-url ws://localhost:8765/ws

Requirements:

- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus import BusBridgeProcessor, BusFrameMessage
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
from pipecat.workers.proxy.websocket import WebSocketProxyClient
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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

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

    # Forward bus frame messages over the WebSocket so the remote
    # assistant sees user-side context and can ship back its replies.
    proxy = WebSocketProxyClient(
        "proxy",
        url=runner_args.cli_args.remote_url,
        local_worker_name=MAIN_NAME,
        remote_worker_name="assistant",
        forward_messages=(BusFrameMessage,),
    )

    async def on_assistant_ready(_data: WorkerReadyData) -> None:
        logger.info("Remote assistant ready, activating")
        await worker.activate_worker(
            "assistant",
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

    await runner.registry.watch("assistant", on_assistant_ready)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected, activating proxy")
        await worker.activate_worker("proxy")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(proxy, worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    parser = argparse.ArgumentParser(description="Main transport worker with WebSocket proxy")
    parser.add_argument(
        "--remote-url",
        default="ws://localhost:8765/ws",
        help="WebSocket URL of the remote LLM server",
    )

    main(parser)
