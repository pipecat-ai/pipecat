#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live (gpt-live-1) with client delegation to an Anthropic backend.

The live model handles the spoken conversation and hands work that needs tools
or careful reasoning to a ``BackendLLMWorker`` running Claude, with its own
context and tools. What the backend says is returned to the live model to
relay to the user.
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
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.llm import BackendLLMWorker
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
or asks you to look something up. The backend reads the conversation, so
hand off as soon as you know the request is for it. While the delegated work
runs, keep the conversation going and relay the result once it arrives;
ignore results the conversation has already moved past.

## Interruptions
Stop speaking when the user interrupts and listen to the new request. If the
user changes an earlier detail, use their latest correction."""

# The backend sees the voice conversation as labelled transcript text; its
# own replies are the only assistant messages in its context.
BACKEND_INSTRUCTIONS = """You are the backend of a voice assistant. Each message you receive
is the recent voice conversation between the user and the assistant, as a
transcript. Work out what is being asked from it and answer that. The
transcript may contain transcription errors; use the most likely intent.

Use the available tools to answer questions about the weather and
restaurants. Reply with the verified result in concise, conversational plain
text that the assistant can say to the user — no Markdown, no raw JSON — and
never claim an action completed without a tool result confirming it."""


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

    # The backend: any LLM service, with its own context and tools. The live
    # service registers it with the runner as a child of the pipeline worker.
    backend = BackendLLMWorker(
        # Reasoning summaries stream back to the frontend as silent context.
        llm=AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                system_instruction=BACKEND_INSTRUCTIONS,
                thinking=AnthropicLLMService.ThinkingConfig(type="adaptive", display="summarized"),
            ),
        ),
        context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
    )

    llm = OpenAILiveLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILiveLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        delegation=OpenAILiveLLMService.ClientDelegation(backend=backend),
    )

    # The live model's own context: no tools here, they belong to the backend.
    context = LLMContext(
        # A trailing developer message asks the model to open the conversation.
        # Comment it out to have the bot wait for the user to speak first.
        [{"role": "developer", "content": "Greet the user and ask how you can help."}],
    )

    # OpenAI Live is full-duplex: it detects the user's turns itself and
    # handles being interrupted, so there is no local VAD and interruptions
    # are never broadcast.
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
    async def on_delegation_created(llm, delegation):
        logger.info(f"Delegated to the backend: {delegation.id}")

    @backend.assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_backend_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"Backend said: {message.content}")

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
