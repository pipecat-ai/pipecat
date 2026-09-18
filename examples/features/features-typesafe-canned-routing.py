#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A voice bot with no LLM: TypeSafe picks which canned line to say.

The bot asks whether the user is over 18 and needs a yes or a no. There is
nothing to generate, so there is no LLM in the pipeline at all. Instead, at
the end of every user turn a ``TypeSafeChoiceRouter`` asks TypeSafe's Jev
model, in one request, which of the known answers the reply matches and
whether the user also asked something. Code then speaks the matching
pre-written line through TTS:

- yes: confirm and end the call
- no: decline and end the call
- the user asked a question, with or without an answer: say we cannot answer
  questions, and ask again
- anything else, or a failed judgment: say we did not catch that, and ask again

Every line the bot says is written here. Jev only chooses between them.

Requirements:
- TYPESAFE_API_KEY
- CARTESIA_API_KEY

Run the example:
uv run examples/features/features-typesafe-canned-routing.py
"""

import os

from dotenv import load_dotenv
from loguru import logger
from typesafe_sdk import Noul

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import EndFrame, Frame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.typesafe_choice_router import TypeSafeChoiceRouter
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.typesafe import JudgeResult, TypeSafeJudge
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
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
    # Behavioral evals: run with `-t eval` to drive this bot via `pipecat eval`.
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}

# Everything the bot can say.
QUESTION = "Are you over 18? Please say yes or no."
CONFIRMED = "Great, you're verified. Goodbye!"
DECLINED = "No problem, you must be 18 or older. Goodbye!"
NO_QUESTIONS = "I can't answer questions right now, but I do need to know: are you over 18?"
UNCLEAR = "Sorry, I didn't catch that. Are you over 18? Please say yes or no."

HAS_QUESTION = "has_question"


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="32b3f3c5-7171-46aa-abe7-b598964aa793",
        ),
    )

    # The context still tracks what was said, so the router can show TypeSafe
    # the bot's question next to the user's reply. No LLM ever reads it.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    async def say(*frames: Frame):
        await worker.queue_frames(frames)

    def asked_something(result: JudgeResult | None) -> bool:
        return result is not None and result.nouls[HAS_QUESTION].probability > 0.5

    async def on_choice(result: JudgeResult, context: LLMContext) -> bool:
        if asked_something(result):
            logger.info("User also asked something; declining to answer and asking again")
            await say(TTSSpeakFrame(NO_QUESTIONS))
            return True
        if result.choices["route"].choice == "yes":
            await say(TTSSpeakFrame(CONFIRMED), EndFrame())
        else:
            await say(TTSSpeakFrame(DECLINED), EndFrame())
        return True

    async def on_fallthrough(result: JudgeResult | None, context: LLMContext) -> bool:
        # Neither a clear yes nor a clear no. A question gets the line that says
        # so; anything else (or TypeSafe being unreachable) gets the re-prompt.
        if asked_something(result):
            await say(TTSSpeakFrame(NO_QUESTIONS))
        else:
            await say(TTSSpeakFrame(UNCLEAR))
        return True

    # The reply is a speech-to-text transcript, and Jev is told so: STT hears
    # "know" for "no" often enough that a plain reading would send a clear
    # "no" to the re-prompt.
    router = TypeSafeChoiceRouter(
        judge=TypeSafeJudge(),
        instructions=(
            "How did the user answer the question in `bot_message`? Judge `user_reply`, which "
            "is a speech-to-text transcript of what the user said, so read it by sound: a "
            "one-word reply that sounds like yes or no is that answer even if it was "
            "transcribed as another word (for example 'know' for 'no')."
        ),
        criteria={
            "yes": "The user confirms they are over 18: yes, yeah, yep, sure, I am, or similar",
            "no": (
                "The user says they are not over 18: no, nope, nah, I'm not, or a transcript "
                "that sounds like no"
            ),
        },
        fallback_description="Neither a yes nor a no: unclear, off topic, a stall, or a question",
        extra_questions={
            HAS_QUESTION: Noul(
                instructions=(
                    "Besides answering, does the user in `user_reply` ask a question or raise "
                    "a concern the bot should respond to before moving on?"
                )
            )
        },
        on_choice=on_choice,
        on_fallthrough=on_fallthrough,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            router,  # Picks the canned line; there is no LLM after it.
            tts,
            transport.output(),
            context_aggregator.assistant(),
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
        logger.info("Client connected")
        await say(TTSSpeakFrame(QUESTION))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

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
