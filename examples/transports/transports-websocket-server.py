#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server transport: the client dials in, the bot listens.

Every other transport example reaches out to a service or serves a browser.
This one is the inverse: the bot opens a WebSocket server and waits for a
client to connect to it. That suits anything you cannot hand a browser to, a
microcontroller or an embedded device, which opens one socket, streams
microphone audio up and plays whatever comes back down.

Two things follow from that shape and are worth knowing before you copy this:

- **A serializer is required.** `SingleClientWebsocketServerParams.serializer` defaults to
  `None`, and with no serializer every inbound message is silently dropped, so
  the bot simply never hears anything. `ProtobufFrameSerializer` is used here;
  a device speaking its own wire format would subclass `FrameSerializer`
  instead.
- **One client at a time.** While a client is connected, new connections are
  refused and the existing one is kept. Good for a single device or local
  development, not for serving many callers at once.

The brain is `gpt-live-1`, which is full-duplex: it hears the user while it is
speaking and decides on its own when to stop, so there is no local VAD in this
pipeline and interruptions are never broadcast.

Run it, then point a client at ``ws://localhost:8765``::

    python examples/transports/transports-websocket-server.py

Required environment: ``OPENAI_API_KEY``.
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

FRONTEND_INSTRUCTIONS = """## Role and speaking style
You are a friendly, concise voice assistant speaking to someone through a
small device. Speak naturally, in one or two sentences at a time.

## Delegation
Answer simple conversational questions directly. Delegate anything that needs
a lookup or a tool. While the delegated work runs, keep the conversation
going and relay the result once it arrives.

## Interruptions
Stop speaking when the user interrupts and listen to the new request."""

BACKEND_INSTRUCTIONS = """You are helping an assistant during a live voice
conversation. The request may contain transcription errors; use the most
likely intent. Use the available tools, and return the result in concise,
conversational plain text with no Markdown and no raw JSON."""


async def set_light(params: FunctionCallParams, room: str, on: bool):
    """Turn a light on or off.

    Args:
        room: Which room, e.g. "office".
        on: True to turn the light on, False to turn it off.
    """
    await params.result_callback({"room": room, "on": on, "ok": True})


async def main():
    # The serializer is not optional in practice: without one, inbound
    # messages are dropped and the bot never hears the client.
    transport = SingleClientWebsocketServerTransport(
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=ProtobufFrameSerializer(),
        ),
        host="localhost",
        port=8765,
    )

    llm = OpenAILiveLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILiveLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        delegation=OpenAILiveLLMService.ResponsesDelegation(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-5.4-mini",
                system_instruction=BACKEND_INSTRUCTIONS,
            ),
        ),
    )

    # The context's tools belong to the backend model; their handlers run here.
    context = LLMContext(
        [{"role": "developer", "content": "Greet the user and ask how you can help."}],
        [set_light],
    )

    # gpt-live-1 detects turns itself and handles being interrupted, so there
    # is no VAD in this pipeline and interruptions are never broadcast.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner()
    await runner.add_workers(worker)

    @transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(transport):
        logger.info("Listening on ws://localhost:8765 — connect a client")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Starting the session from the context is what makes the bot speak
        # first; drop this to have it wait for the user instead.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")

    await runner.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
