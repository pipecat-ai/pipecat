#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live (gpt-live-1) where the backend decides which updates are spoken.

``transform_output`` sets the ``speakable`` flag on each thing the backend
produces. Here the backend marks the messages it wants heard and the transform
reads that convention; an app can decide any other way it likes. Marked
messages are relayed aloud, and the rest — notes to self, reasoning summaries,
the final wrap-up — stays silent context the live model can draw on if asked.

This suits a backend that works for a while and narrates its own progress.
"""

import os
from dataclasses import replace
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
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.llm import BackendLLMWorker, BackendOutput
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

#: Prefix the backend puts on anything it wants the user to hear.
SPEAK_MARKER = ">>"

FRONTEND_INSTRUCTIONS = """## Role and speaking style
You are a friendly, concise voice assistant. Speak naturally, in one or two
sentences at a time, and let the user finish before responding.

## Delegation
Answer simple conversational questions directly. Delegate when the user asks
for current information, such as the weather or a restaurant recommendation,
or asks you to look something up. The backend reads the conversation, so hand
off as soon as you know the request is for it.

The backend chooses what the user should hear and sends it to you as it
works; relay those updates as they arrive. Everything else it does reaches
you as context you know but need not repeat.

## Interruptions
Stop speaking when the user interrupts and listen to the new request."""

BACKEND_INSTRUCTIONS = f"""You are the backend of a voice assistant. Each message you receive
is the recent voice conversation between the user and the assistant, as a
transcript. Work out what is being asked from it and answer that. The
transcript may contain transcription errors; use the most likely intent.

You decide what the user hears. Begin a message with {SPEAK_MARKER} and the
whole of it is said to them; write anything else and the whole message stays a
note to yourself. Speak up when you have something worth hearing — the
verified result, or a word about what is taking time — and keep a spoken
message to one or two sentences. Work out loud as much as you like in the
messages you leave unmarked.

Use the available tools to answer questions about the weather and
restaurants. Never claim an action completed without a tool result confirming
it."""


async def transform_output(output: BackendOutput) -> BackendOutput:
    """Let the backend's own marker decide what reaches the user."""
    # `speakable` is one flag on one output, and an output is a whole message
    # the backend wrote, so the marker opens the message it applies to.
    if output.text.startswith(SPEAK_MARKER):
        return replace(output, text=output.text[len(SPEAK_MARKER) :].lstrip(), speakable=True)
    return replace(output, speakable=False)


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

    backend = BackendLLMWorker(
        llm=AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                system_instruction=BACKEND_INSTRUCTIONS,
                thinking=AnthropicLLMService.ThinkingConfig(type="adaptive", display="summarized"),
            ),
        ),
        context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
        transform_output=transform_output,
    )

    llm = OpenAILiveLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILiveLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        delegation=OpenAILiveLLMService.ClientDelegation(backend=backend),
    )

    context = LLMContext(
        [{"role": "developer", "content": "Greet the user and ask how you can help."}],
    )

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

    @backend.assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_backend_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        spoken = "spoken" if message.content.startswith(SPEAK_MARKER) else "silent"
        logger.info(f"Backend ({spoken}): {message.content}")

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
