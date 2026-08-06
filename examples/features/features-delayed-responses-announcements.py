#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example demonstrating announcement wording and staying silent.

``DelayedResponseStrategy`` decides *when* a finished background task is
announced (see ``features-delayed-responses.py``). This example covers the
other half: *how much the bot says* when that moment arrives — an
application choice, because a task finishing mid-conversation is an
interruption, and how much of one it should be depends on the app.

Two tools, two answers:

- ``research`` announces itself with the application's own wording, via
  ``AnnouncementConfig(single_prompt=...)``. The shipped instruction is
  replaced, so the announcement sounds like this app rather than like the
  framework default.
- ``watch_price`` never interrupts at all. It delivers its final result with
  ``FunctionCallResultProperties(run_llm=False)``, so nothing is announced —
  but the result still lands in the LLM context, and asking "did that price
  ever move?" later gets a real answer. Silence belongs to the tool, which
  is the only place that knows whether a result was worth interrupting for.

Things to try in a conversation:

- Ask for research, then keep chatting: the announcement arrives in the
  app's wording when there's an opening.
- Ask it to watch a price, then keep chatting: nothing ever interrupts you.
  Ask about it later and the answer is there.

Run with::

    uv run examples/features/features-delayed-responses-announcements.py
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import FunctionCallResultProperties, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.response import AnnouncementConfig, DelayedResponseStrategy
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will be "
    "spoken aloud, so avoid emojis, bullet points, or other formatting that can't "
    "be spoken. Keep responses brief. Whenever the user asks you to research, "
    "look into, or find out about a topic, call the research tool — do not answer "
    "from memory. Whenever the user asks you to watch, track, or keep an eye on a "
    "price, call the watch_price tool — never say you are watching a price without "
    "calling it. Both tools run in the background: after starting one, tell the user "
    "you'll let them know when there's something to report, and keep conversing "
    "normally in the meantime."
)


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def research(params: FunctionCallParams, topic: str):
    """Research a topic. Takes a while; the result is announced when it's ready.

    Args:
        topic: The topic to research, e.g. "the Mariana Trench".
    """
    await params.result_callback(
        {"status": "started", "topic": topic},
        properties=FunctionCallResultProperties(is_final=False, run_llm=True),
    )

    # Stand-in for real long-running work (an API call, a sub-agent, ...).
    await asyncio.sleep(8)
    logger.info(f"Research on {topic} is complete.")
    finding = (
        f"Research on {topic} is complete. Key finding: 3 sources agree, "
        "with a confidence score of 0.82."
    )

    await params.result_callback({"status": "complete", "finding": finding})


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def watch_price(params: FunctionCallParams, item: str):
    """Watch an item's price in the background. Results are not announced.

    Args:
        item: The item to watch, e.g. "the blue hiking boots".
    """
    await params.result_callback(
        {"status": "watching", "item": item},
        properties=FunctionCallResultProperties(is_final=False, run_llm=True),
    )

    # Stand-in for a long-running watch (a polling loop, a webhook, ...).
    await asyncio.sleep(8)
    logger.info(f"Price check on {item} is complete.")
    reading = f"The price of {item} dropped 12 percent, to 44 dollars."

    # run_llm=False keeps this out of the conversation: the reading lands in
    # the LLM context so a later question can be answered from it, but
    # nothing is announced and no inference runs.
    await params.result_callback(
        {"status": "complete", "reading": reading},
        properties=FunctionCallResultProperties(run_llm=False),
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
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    # The announcement is worded by this app rather than by the shipped
    # default. Both cases keep AnnouncementStyle.RESULT — a custom prompt
    # replaces whatever instruction the style would have supplied.
    announcement = AnnouncementConfig(
        single_prompt=(
            "The background task '{name}' you started earlier has finished, and its "
            'result is already in the conversation. Begin your reply with "Heads up:" '
            "and then briefly state what it found. Refer to the task the way it came "
            "up in conversation, not by its function name."
        ),
        multiple_prompt=(
            "{count} background tasks you started earlier have finished: {names}. "
            'Their results are already in the conversation. Begin your reply with "Heads up:" '
            "and then briefly state what each one found. Refer to the tasks the way they "
            "came up in conversation, not by their function names."
        ),
    )

    context = LLMContext(tools=[research, watch_price])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        assistant_params=LLMAssistantAggregatorParams(
            response_strategy=DelayedResponseStrategy(
                settle_secs=2.0,
                announcement=announcement,
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
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
