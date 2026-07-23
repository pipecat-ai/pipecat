#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The food ordering flow driven by a speech-to-speech (realtime) model.

Same conversation flow as food_ordering.py — the node definitions are imported
from there — but the LLM is OpenAI Realtime (``gpt-realtime``): audio in, audio
out, server-side turn detection, and no separate STT/TTS services. Used to
evaluate Pipecat Flows against realtime models (see scripts/release-evals);
drive it with the audio-mode ``food_ordering_pizza_realtime`` scenario — the
service doesn't implement text-driven turns yet, so text-mode scenarios don't
apply.

Requirements:
- OPENAI_API_KEY
"""

import asyncio
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from food_ordering import DeliveryEstimateResult, create_initial_node
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import FlowManager, flows_tool_options
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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
    """Run the food ordering bot on a realtime (speech-to-speech) LLM."""
    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            session_properties=SessionProperties(
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(),
                        turn_detection=SemanticTurnDetection(),
                    )
                ),
            ),
        ),
    )

    # The realtime service emits its own user-turn frames from server VAD, so
    # no local VAD on the aggregator; realtime-service mode is auto-detected.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
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

    # Define "global" functions available at every node
    @flows_tool_options(cancel_on_interruption=True)
    async def get_delivery_estimate(
        flow_manager: FlowManager,
    ) -> tuple[DeliveryEstimateResult, None]:
        """Provide delivery estimate information. Only call this when the user explicitly asks about delivery timing; never call it proactively."""
        # Simulated slow lookup; with cancel_on_interruption a user barge-in
        # abandons it (the release evals exercise both the slow completion and
        # the barge-in cancellation).
        await asyncio.sleep(4)
        delivery_time = datetime.now() + timedelta(minutes=30)
        return DeliveryEstimateResult(
            time=f"{delivery_time}",
        ), None

    @flows_tool_options(cancel_on_interruption=True, timeout_secs=3)
    async def check_daily_special(
        flow_manager: FlowManager,
    ) -> tuple[dict, None]:
        """Look up today's special menu item. Only call this when the user explicitly asks about today's special; never call it proactively."""
        # Simulated hung backend: never completes, so the per-tool timeout
        # resolves the call instead (exercised by the release evals).
        await asyncio.sleep(3600)
        return {"special": "unavailable"}, None

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=[get_delivery_estimate, check_daily_special],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation with the initial node
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
