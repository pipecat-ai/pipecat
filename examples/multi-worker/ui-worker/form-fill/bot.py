#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Form-fill: a voice-guided, accessible form walkthrough.

An accessibility-oriented take on form filling: instead of waiting for
the user to dictate values, the assistant leads. It walks the user through
a job application one section at a time, personal information (name,
email, phone), then job qualifications (years of experience and why they
are interested), then submit, confirming what it captured before moving
on. A user who cannot see the screen never has to.

The voice LLM leads the whole conversation and works the form through
``UIWorker``'s one screen tool: ``screen("list", "textbox")`` shows it
which inputs are filled and which are still empty, ``screen("fill", "the
email field", value)`` writes a value into the field the user means, and
``screen("click", "the submit button")`` submits. The UI worker finds each
field with its classifier and sends the command; no LLM turn runs on the
UI side, and the voice LLM never sees the page.

The flow is driven off the form itself: each turn the voice LLM lists the
inputs and steers toward the next empty one, so progress is the form, not
hidden conversation state.

The worker's classifier is its own LLM through an ``LLMClassifier``; pass a
``JevClassifier`` for faster, calibrated answers.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in -> STT -> user_agg -> LLM -> TTS -> transport.out -> assistant_agg
        └── screen_tools("ui"): screen(action, target, value)
              └── params.pipeline_worker.job("ui", name="screen", payload=...)

    UIWorker ("ui", with a classifier, no LLM turn):
      └── built-in "screen" job -> "list": the inputs with their values
                                   "fill" / "click": classifier finds the field, sends the command

Run::

    uv run bot.py

Then open the client at ``http://localhost:5173`` (see ``README.md``).

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner
from pipecat.workers.ui import UIWorker, screen_tools

load_dotenv(override=True)

MAIN_NAME = "main"
UI_NAME = "ui"

transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


VOICE_PROMPT = """\
You are a warm, patient assistant helping the user fill out a job \
application entirely by voice. The user cannot see the screen, so YOU \
lead: ask for each piece of information, write it into the form, and \
tell the user what you captured before moving on. You cannot see the \
screen either; your tools work the form for you.

## The flow, in order

1. Personal information: first name, last name, email, phone number.
2. Job qualifications: years of relevant experience, and why they are \
interested in the role.
3. Submit.

## The screen tool

- screen(action="list", target="textbox"): the form's inputs with their \
current values. Call it at the start of every turn to see which fields \
are filled and steer toward the next empty one in the current step.
- screen(action="fill", target=..., value=...): write one value into the \
field the target names, such as "the email field". Call it once per \
value; several in one turn is fine.
- screen(action="click", target="the submit button"): submit, at the \
very end.

## How to guide

- User gives one or more values: write them, confirm briefly ("Got it, \
John Smith"), and ask for the next missing item in the current step.
- A step is complete: say so and move to the next step's first field.
- Everything is filled: say the form is complete and ask if they are \
ready to submit. Do not read the values back; each one was confirmed \
when captured.
- User says to submit: click the submit button and give a short send-off \
only ("Submitting your application now, good luck!"). Nothing after.
- User corrects a value: write it again and confirm the change.
- A tool answers that a field was not found: say so and ask again.

Ask for one thing at a time (a full name counts as one thing). Keep \
every reply to one or two short spoken sentences.

## Spelling

Convert spoken forms to the stored value: "john at example dot com" is \
john@example.com; "five five five one two three four" is 5551234; "five \
years" is 5. Confirm naturally ("got it, your email's john@example.com")."""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting form-fill bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "86e30c1d-714b-4074-a1f2-1cb6b552fb49"),
        ),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(system_instruction=VOICE_PROMPT),
    )

    context = LLMContext(tools=screen_tools(UI_NAME))
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    ui_worker = UIWorker(UI_NAME, llm=OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"]))

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(ui_worker, worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user warmly. In one or two short sentences, tell "
                    "them you'll guide them through this job application by voice, "
                    "one step at a time, and ask for their name to begin."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
