#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Talk to the Zoom Voice Agent, with function calling.

The Voice Agent runs the whole voice loop on Zoom's side: it detects turns,
transcribes the caller, runs the model and speaks the reply. Set
``ZOOM_AI_API_KEY`` to a Zoom AI Services API key from Zoom Platform Studio.

Requirements:
    uv sync --extra zoom-ai --extra daily --extra livekit --extra runner \
        --extra websocket --extra webrtc

Usage:
    uv run python examples/realtime/realtime-zoom-ai.py -t webrtc
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.zoom_ai.realtime.events import Speak
from pipecat.services.zoom_ai.realtime.llm import ZoomAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = 75 if format == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback({"name": "The Golden Dragon"})


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

    llm = ZoomAIRealtimeLLMService(
        api_key=os.environ["ZOOM_AI_API_KEY"],
        settings=ZoomAIRealtimeLLMService.Settings(
            system_instruction="""You are a helpful and friendly AI in a voice conversation.

Your voice and personality should be warm and engaging. Keep your responses short: one or two
sentences, usually. When a tool can answer, call it right away, without saying that you will.
Respond in English.""",
            speak=Speak(voice="jenna"),
        ),
    )

    # The Voice Agent converts the standard messages format of the context to
    # its own conversation items.
    context = LLMContext(
        [{"role": "developer", "content": "Say hello!"}],
        [get_current_weather, get_restaurant_recommendation],
    )

    # The Voice Agent detects turns and handles barge-in itself, so the
    # aggregators need no local VAD.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            user_aggregator,
            llm,  # Zoom Voice Agent
            transport.output(),  # Transport bot output
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[TranscriptionLogObserver()],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        await worker.queue_frames([LLMRunFrame()])

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
