#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live (gpt-live-1) with an app deciding what the user hears from its backend.

A ``BackendLLMWorker``'s model decides which of what it writes the user hears:
it speaks up for a result, a question it cannot proceed without, and news the
user should have now, and works through the steps in between in silence. An
app has two levers over that.

What must be said, a tool says itself, with ``send_output``. ``rebook_flight``
tells the user the new flight, seat and confirmation code as the booking
system gave them: spoken whatever the model would have chosen, and in no one's
paraphrase. Its result says the user has been told.

The backend's prompt shapes the model's judgment, in plain words. The last
paragraph of ``BACKEND_INSTRUCTIONS`` says the booking system announces a
booking itself, so the model never repeats it and adds only what the user
still needs to know, in a sentence or nothing. The model still speaks up on
its own for news, as when it finds the flight cancelled: the prompt steers
its judgment rather than replacing it.

``transform_output`` is the third lever, for code that should see every
output the model produces: it can drop one, rewrite it, or change whether it
is spoken. Here it only logs each one with its flag.

To hear it, ask for something that takes the backend several steps, like "my
flight UA482 this morning — can you check it, and get me on something else if
it's not running?" Any flight number does: the tools report that one cancelled
whatever you give them.
"""

import asyncio
import os

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
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.llm import BackendLLMWorker, BackendOutput
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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

The backend speaks up when it has something for the user; relay that.
Everything else it does reaches you as context: keep it to yourself, and draw
on it only if the user asks what is happening.

## Interruptions
Stop speaking when the user interrupts and listen to the new request."""

BACKEND_INSTRUCTIONS = """You look after the user's flights and bookings: checking a flight,
finding another, moving a booking. The conversation you are sent may contain
transcription errors; use the most likely intent.

See a request through. When one takes several steps, as rebooking does (check
the flight, find what else flies the route, book a seat on the earliest flight
that has them), carry on to the end rather than coming back with a menu. Never
claim an action completed without a tool result confirming it.

The booking system tells the user itself when a booking goes through, with
the flight, seat and confirmation code, so never repeat those; once it has,
add only what they still need to know, in a sentence, or nothing. If a job
ends without a booking, tell them once where they stand."""


async def log_output(output: BackendOutput) -> BackendOutput:
    """Log each output with the flag the model's choice gave it, and send it on as is."""
    kind = "thought" if output.is_thought else "spoken" if output.prefers_spoken else "note"
    logger.info(f"Backend ({kind}): {output.text}")
    return output


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
    booking = {"flight": flight_number, "departs": "16:15", "seat": "14C", "confirmation": "X7K2QP"}
    # The confirmation is told to the user by the tool itself, as the booking
    # system gave it, whatever the model would have chosen to say.
    told = (
        f"You're booked on {booking['flight']}, departing at {booking['departs']}, "
        f"seat {booking['seat']}. Your confirmation code is {booking['confirmation']}."
    )
    backend = params.pipeline_worker
    assert isinstance(backend, BackendLLMWorker)
    await backend.send_output(BackendOutput(text=told))
    await params.result_callback({**booking, "status": "confirmed", "user_told": told})


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

    backend = BackendLLMWorker(
        llm=AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                system_instruction=BACKEND_INSTRUCTIONS,
                thinking=AnthropicLLMService.ThinkingConfig(type="adaptive", display="summarized"),
            ),
        ),
        context=LLMContext(tools=[check_flight_status, find_alternative_flights, rebook_flight]),
        transform_output=log_output,
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
