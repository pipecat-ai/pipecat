#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example demonstrating ``DelayedResponseStrategy`` for assistant-initiated speech.

A bot's reactive answers are timed by the user turn system: the user stops
speaking, the bot answers. But *assistant-initiated* speech — content the
conversation didn't just ask for, like the result of a long-running task
finishing while the user is mid-sentence — has no natural slot. Pushed raw,
it lands on top of whatever is happening.

This example wires the two pieces that solve that:

- The ``research`` tool is a plain async tool
  (``@tool_options(cancel_on_interruption=False)``): it acknowledges
  immediately with an intermediate result, keeps running — a fixed 15 s
  sleep standing in for real work — while the conversation continues, and
  simply reports its final result. Nothing announcement-specific in the tool.
- The ``DelayedResponseStrategy`` configured on the assistant aggregator does
  the rest: an async tool's completion routes through it natively, so the
  announcement waits for a conversational opening — bot quiet, user quiet, no
  answer owed, settle window elapsed — then releases as a single LLM run.
  Results that complete close together release as one batch, announced by one
  composed message (one spoken response, not a chain). The phrasing of that
  announcement is configurable via ``DelayedResponseStrategy(announcement=...)``
  — e.g. tease with "the result for X is ready, want to hear it?" instead of
  announcing outright.

(App code can also schedule assistant-initiated speech from non-tool sources
— timers, external events — by queueing a ``ResponseFrame``; the strategy
treats both the same way.)

The strategy is configuration on the assistant aggregator: a plain
single-pipeline bot, with no extra frame processors.

Things to try in a conversation:

- Ask for research, then keep chatting: the announcement waits for a pause
  instead of talking over you.
- Ask for research on two topics back to back: completions arriving close
  together are announced once, in one response.
- Ask for research, then ask an unrelated question right as it completes:
  the announcement waits until your question has been answered.

Run with::

    uv run examples/features/features-delayed-responses.py
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
from pipecat.turns.response import DelayedResponseStrategy
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will be "
    "spoken aloud, so avoid emojis, bullet points, or other formatting that can't "
    "be spoken. Keep responses brief. Whenever the user asks you to research, "
    "look into, or find out about a topic, call the research tool — do not answer "
    "from memory. It runs in the background: after starting it, tell the user "
    "you'll let them know when the results are in, and keep conversing normally "
    "in the meantime."
)


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def research(params: FunctionCallParams, topic: str):
    """Research a topic. Takes a while; the result is announced when it's ready.

    Args:
        topic: The topic to research, e.g. "the Mariana Trench".
    """
    # Acknowledge immediately with an intermediate result so the bot can tell
    # the user the research has started. cancel_on_interruption=False (via
    # @tool_options above) makes this an async tool: the conversation
    # continues while the work below runs, and the call survives
    # interruptions.
    await params.result_callback(
        {"status": "started", "topic": topic},
        properties=FunctionCallResultProperties(is_final=False, run_llm=True),
    )

    # Stand-in for real long-running work (an API call, a sub-agent, ...).
    await asyncio.sleep(15)
    logger.info(f"Research on {topic} is complete.")
    finding = (
        f"Research on {topic} is complete. Key finding: 3 sources agree, "
        "with a confidence score of 0.82."
    )

    # That's the whole tool: because this is an async call and a response
    # strategy is configured, the DelayedResponseStrategy schedules the
    # announcement for the next conversational opening — the result lands in
    # context now, and the bot speaks it when there's a polite moment.
    await params.result_callback({"status": "complete", "finding": finding})


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

    context = LLMContext(tools=[research])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        assistant_params=LLMAssistantAggregatorParams(
            response_strategy=DelayedResponseStrategy(settle_secs=2.0),
        ),
    )

    @assistant_aggregator.event_handler("on_response_deferred")
    async def on_response_deferred(aggregator, frame):
        logger.info(f"Assistant-initiated response deferred: {frame}")

    @assistant_aggregator.event_handler("on_response_released")
    async def on_response_released(aggregator, frames):
        logger.info(f"Assistant-initiated responses released: {len(frames)}")

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
