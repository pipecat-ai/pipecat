#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live (gpt-live-1) where the backend decides which updates are spoken.

Rebooking a cancelled flight takes the backend three tool calls, seconds
apart, each needing the last one's answer. It works through them in unmarked
messages and marks the two the user is waiting to hear: that the flight is
gone and it is looking for another, then the seat it found. A caller that
spoke every update would put the backend's whole working-out through the live
model's voice.

To hear it, ask for something like "my flight UA482 this morning — can you
check it, and get me on something else if it's not running?" Any flight
number does: the tools report that one cancelled whatever you give them.

A marked message is relayed; an unmarked one becomes thinking context, which
the live model is not asked to say but may still work into what it says.

``transform_output`` reads that marker and sets ``speakable``. A backend can
say the same thing by calling a tool to talk to the user; a marker convention
needs no extra plumbing, and ``speakable`` is what the live model acts on
either way.
"""

import asyncio
import os
from dataclasses import replace

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
Answer simple conversational questions directly. Delegate anything about the
user's flights or bookings — checking one, changing one, finding another. The
backend reads the conversation, so hand off as soon as you know the request is
for it.

The backend chooses what the user should hear and sends it to you as it
works; relay those updates as they arrive. Everything else it does reaches
you as context: keep it to yourself, and draw on it only if the user asks
what is happening.

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

Rebooking runs in steps: check the flight, find what else flies that route,
then book a seat on the earliest one that has them. Each step needs the one
before it, and the user has asked you to see it through, so carry on to the
booking rather than coming back with a menu. Say something when the user
would otherwise be waiting with no news — when you learn something that
changes their plans, and when the job is done. Never claim an action
completed without a tool result confirming it."""


async def transform_output(output: BackendOutput) -> BackendOutput:
    """Let the backend's own marker decide what reaches the user."""
    # `speakable` is one flag on one output, and an output is a whole message
    # the backend wrote, so the marker opens the message it applies to.
    if output.text.startswith(SPEAK_MARKER):
        return replace(output, text=output.text[len(SPEAK_MARKER) :].lstrip(), speakable=True)
    return replace(output, speakable=False)


# The three steps of a rebooking, each slow enough that the user notices the
# wait, and each needing what the step before it returned.


async def check_flight_status(params: FunctionCallParams, flight_number: str):
    """Check whether a flight is running, and what route it flies.

    Args:
        flight_number: The flight number, e.g. "UA482".
    """
    await asyncio.sleep(3)
    await params.result_callback(
        {
            "flight": flight_number,
            "status": "cancelled",
            "reason": "crew shortage",
            "origin": "SFO",
            "destination": "JFK",
            "scheduled_departure": "11:40",
        }
    )


async def find_alternative_flights(params: FunctionCallParams, origin: str, destination: str):
    """Find later flights on a route today.

    Args:
        origin: Departure airport code, e.g. "SFO".
        destination: Arrival airport code, e.g. "JFK".
    """
    await asyncio.sleep(4)
    await params.result_callback(
        {
            "flights": [
                {"flight": "UA716", "departs": "16:15", "seats": 2},
                {"flight": "AA229", "departs": "19:05", "seats": 11},
            ]
        }
    )


async def rebook_flight(params: FunctionCallParams, flight_number: str):
    """Move the booking onto another flight.

    Args:
        flight_number: The flight to move the booking to, e.g. "UA716".
    """
    await asyncio.sleep(3)
    await params.result_callback(
        {"flight": flight_number, "status": "confirmed", "seat": "14C", "confirmation": "X7K2QP"}
    )


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
        context=LLMContext(tools=[check_flight_status, find_alternative_flights, rebook_flight]),
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
