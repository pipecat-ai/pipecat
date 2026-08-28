#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live (gpt-live-1) with Responses delegation.

The live model handles the spoken conversation and hands work that needs tools
or careful reasoning to an OpenAI-hosted Responses model. The backend's
function calls run here, with the handlers registered for the tools in the
``LLMContext``.
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
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# The live model only needs to know how to converse and when to delegate. Task
# knowledge, tools and business rules belong to the backend prompt.
FRONTEND_INSTRUCTIONS = """## Role and speaking style
You are a friendly, concise voice assistant. Speak naturally, in one or two
sentences at a time, and let the user finish before responding.

## Delegation
Answer simple conversational questions directly. Delegate when the user asks
for current information, such as the weather or a restaurant recommendation,
or asks you to look something up. When delegating, include the user's goal,
the exact details they gave (places, dates, names) and their latest
correction, so the request is self-contained. While the delegated work runs,
keep the conversation going and relay the result once it arrives; ignore
results the conversation has already moved past.

## Interruptions
Stop speaking when the user interrupts and listen to the new request. If the
user changes an earlier detail, use their latest correction."""

BACKEND_INSTRUCTIONS = """You are helping an assistant during a live voice
conversation. The request may contain transcription errors; use the most
likely intent. Use the available tools to answer questions about the weather
and restaurants. Return the verified result in concise, conversational plain
text — no Markdown, no raw JSON — and never claim an action completed without
a tool result confirming it."""


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

    llm = OpenAILiveLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILiveLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        delegation=OpenAILiveLLMService.ResponsesDelegation(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-5.4-mini",
                system_instruction=BACKEND_INSTRUCTIONS,
                reasoning=OpenAIResponsesLLMService.ReasoningConfig(effort="low"),
            ),
        ),
    )

    # The context's tools are the backend model's tools; their handlers run
    # here. The trailing developer message seeds the session so the model
    # speaks first.
    context = LLMContext(
        [{"role": "developer", "content": "Greet the user and ask how you can help."}],
        [get_current_weather, get_restaurant_recommendation],
    )

    # OpenAI Live is full-duplex: it detects the user's turns itself and
    # handles being interrupted, so there is no local VAD and interruptions
    # are never broadcast. Realtime-service mode is auto-detected.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            user_aggregator,
            llm,  # LLM
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
        # Start the Live session from the context.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    @llm.event_handler("on_delegation_created")
    async def on_delegation_created(llm, item):
        logger.info(f"Delegated to the backend: {item.text}")

    # In realtime mode the user message is written to the context when the
    # assistant responds, so subscribe to on_user_turn_message_added for the
    # finalized user text.
    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        logger.info(f"Transcript: user: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"Transcript: assistant: {message.content}")

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
