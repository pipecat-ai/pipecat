#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice agent with live web search via Keenable.

Adds low-latency web search to a voice agent using ``KeenableWebSearch``, which
exposes a single ``search_web`` function tool backed by a hosted MCP server
powered by Keenable AI (https://keenable.ai). Pass ``search.tools()`` into the
``LLMContext`` and the LLM auto-registers the tool's handler, so the agent can
answer questions about current events and anything beyond the model's training
data.

No API key is required — the server works keyless by default. Pass an API key
(via ``KEENABLE_API_KEY`` here) for higher rate limits and the lower-latency
``realtime`` mode (a good fit for voice), selected with ``mode="realtime"``.
"""

import os
from datetime import date

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
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
from pipecat.services.keenable.search import KeenableWebSearch
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    system_prompt = f"""\
You are a helpful assistant in a voice conversation with live web access.
Today's date is {date.today():%A, %B %d, %Y}.

You have two tools:
- search_web: search the web for current events, news, or any facts that may be beyond your training data or need to be up to date.
- fetch_page: read the text of a specific web page when the user gives you a URL.

Prefer these tools over guessing or relying on memory whenever a question needs current information. When the user gives you a URL, read it with fetch_page even if you think you already know its contents.

Your output will be spoken aloud, so avoid emojis, URLs, bullet points, or other formatting that can't easily be spoken. Don't overexplain what you are doing; respond with short sentences.
"""

    llm = OpenAIResponsesLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesLLMService.Settings(
            system_instruction=system_prompt,
        ),
    )

    # Keyless by default (uses the "pro" search mode). Pass an API key for higher
    # rate limits and the lower-latency "realtime" mode (best fit for voice;
    # requires an enabled account), selected with mode="realtime".
    # The search tools carry their handlers, so the LLM auto-registers them from
    # the context's tools — no register() call needed. The MCP connection opens
    # lazily on the first call.
    search = KeenableWebSearch(api_key=os.getenv("KEENABLE_API_KEY"))

    context = LLMContext(tools=search.tools())
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
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await search.close()
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
