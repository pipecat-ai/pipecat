#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A 'Hello-World' introduction to Pipecat Flows, with the flow in YAML.

The same bot as python/hello_world.py, split along the seam Pipecat Flows offers for
runtime configuration: flow.yaml holds the two nodes and the transition
between them, and handlers.py holds the one tool the flow calls.

Requirements:
- CARTESIA_API_KEY
- GOOGLE_API_KEY

Run the example:
uv run bot.py
"""

import os
from pathlib import Path

import handlers
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import Flow, FlowConfig, FlowManager
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

FLOW_CONFIG_PATH = Path(__file__).with_name("flow.yaml")

transport_params = {
    "daily": lambda: DailyParams(
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
    # Behavioral evals: run with `-t eval` to drive this bot via `pipecat eval`.
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="32b3f3c5-7171-46aa-abe7-b598964aa793",
        ),
    )
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY", ""))

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            filter_incomplete_user_turns=True,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    # The flow is data: load it, then join it to the tool it names.
    config = FlowConfig.from_file(FLOW_CONFIG_PATH)
    flow = Flow(config, handlers=handlers)

    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=flow.global_functions,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await flow_manager.initialize(flow.initial_node)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
