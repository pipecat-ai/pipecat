#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live (gpt-live-1) with the app's own rule for which backend output is spoken.

A ``BackendLLMWorker`` has a default rule: what its model writes in a turn
with no tool calls is flagged to be spoken, and the model is told so. If you
can describe to the model a better way to choose what the user hears, replace
the rule. Two arguments do it:

- ``output_instructions`` tells the model your rule. Here: mark what the user
  should hear.
- ``transform_output`` sets ``prefers_spoken`` by it. Here: from the mark.

A marked message is relayed; an unmarked one becomes thinking context, which
the live model is not asked to say but may still work into what it says.

To hear it, ask for something that takes the backend several steps, like "my
flight UA482 this morning — can you check it, and get me on something else if
it's not running?" Any flight number does: the tools report that one cancelled
whatever you give them.
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

#: What the backend's model puts at the start of a message the user should hear.
SPEAK_MARKER = ">>"

#: What to say to set the backend's three-step job going.
SUGGESTED_REQUEST = (
    "My flight UA482 this morning — can you check it, and get me on something "
    "else if it's not running?"
)

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

BACKEND_INSTRUCTIONS = """You look after the user's flights and bookings: checking a flight,
finding another, moving a booking. The conversation you are sent may contain
transcription errors; use the most likely intent.

See a request through. When one takes several steps, as rebooking does (check
the flight, find what else flies the route, book a seat on the earliest flight
that has them), carry on to the end rather than coming back with a menu. Never
claim an action completed without a tool result confirming it."""

#: The app's rule for what the user hears, given to the backend's model in
#: place of the worker's default one. ``mark_decides_speech`` is its other half.
OUTPUT_INSTRUCTIONS = f"""WHAT THE USER HEARS: You choose, message by message. Begin what you
write with {SPEAK_MARKER} and the whole of it is told to the user, whether or not
you call tools in the same turn. Anything you write without the mark is a note
to yourself and is never told to them. Mark what is worth their ear: news that
changes their plans, the outcome when you are done, a question you cannot
proceed without. Say something when they would otherwise be waiting with no
news; they need not hear the steps in between. Keep a marked message to one or
two sentences, in plain text the assistant can speak from: no Markdown, no raw
JSON."""


async def mark_decides_speech(output: BackendOutput) -> BackendOutput:
    """Flag an output for speech when the model marked it, and for nothing else."""
    spoken = not output.is_thought and output.text.startswith(SPEAK_MARKER)
    text = output.text[len(SPEAK_MARKER) :].lstrip() if spoken else output.text
    kind = "thought" if output.is_thought else "spoken" if spoken else "note"
    logger.info(f"Backend ({kind}): {text}")
    return replace(output, text=text, prefers_spoken=spoken)


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
        transform_output=mark_decides_speech,
        output_instructions=OUTPUT_INSTRUCTIONS,
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
        logger.opt(colors=True).info(
            f'<yellow><bold>▶ Say something like:</bold> "{SUGGESTED_REQUEST}"</yellow>'
        )
        # Start the Live session from the context.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

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
