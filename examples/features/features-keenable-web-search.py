#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice agent with live web search via Keenable.

Adds low-latency web search to a voice agent using ``KeenableWebSearch``, a thin
wrapper around Pipecat's ``MCPClient`` that connects to a hosted MCP server
powered by Keenable AI (https://keenable.ai). Search tools are discovered from
the server and registered on the LLM, so the agent can answer questions about
current events and anything beyond the model's training data.

No API key is required — the server works keyless by default. Set
``KEENABLE_API_KEY`` for higher rate limits, and ``KEENABLE_SEARCH_MODE`` to
``realtime`` for the lowest latency (a good fit for voice).

Install: ``pip install "pipecat-ai[keenable,cartesia,deepgram,google,silero]"``
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.keenable.search import KeenableWebSearch
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.transports.websocket.server import WebsocketServerParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: WebsocketServerParams(
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
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    system_prompt = """\
You are a helpful LLM in a voice call with live web search.
Use the search tools when asked about current events, facts, or anything beyond your training data.
Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis, URLs, or bullet points.
Don't overexplain what you are doing.
Just respond with short sentences when you are carrying out tool calls.
"""

    llm = GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            system_instruction=system_prompt,
        ),
    )

    # Keyless by default. Set KEENABLE_API_KEY for higher rate limits, or
    # KEENABLE_SEARCH_MODE=realtime for the lowest latency (best fit for voice).
    async with KeenableWebSearch() as search:
        tools = await search.register_tools(llm)

        context = LLMContext(
            messages=[{"role": "user", "content": "Please introduce yourself."}],
            tools=tools,
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,
                user_aggregator,  # User spoken responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                assistant_aggregator,  # Assistant spoken responses and tool context
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

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected: {client}")
            # Kick off the conversation.
            await worker.queue_frames([LLMRunFrame()])

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
