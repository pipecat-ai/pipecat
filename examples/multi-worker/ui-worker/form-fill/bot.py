#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Form-fill — a voice-guided, accessible form walkthrough.

An accessibility-oriented take on form filling: instead of waiting for
the user to dictate values, the assistant *leads*. It walks the user
through a job application one section at a time — personal information
(name, email, phone), then job qualifications (years of experience and
why they're interested), then submit — confirming what it captured
before moving on. A user who can't see the screen never has to; the
assistant asks for each piece, writes it into the form, and reads back
what it heard.

``FormWorker`` composes ``ReplyToolMixin``: the
``reply(answer, scroll_to, fills, click)`` bundle covers the
state-changing actions — ``fills`` writes input values (many at once),
``click`` presses submit. Because the mixin replies with verbatim TTS
(``tts_speak=True``), the worker authors every spoken line, so all the
guidance lives in one place (``UI_PROMPT``).

The flow is driven *statelessly* off ``<ui_state>``: each turn the
worker sees which fields are already filled and steers toward the next
empty one — progress is the form itself, not hidden conversation state.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── answer_about_screen(query) tool
              └── params.pipeline_worker.job("ui", name="respond", payload={query})

    FormWorker (ReplyToolMixin + UIWorker):
      └── inherited: reply(answer, scroll_to, fills, click) — guides the flow

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

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.job_context import JobError
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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner
from pipecat.workers.ui import ReplyToolMixin, UIWorker

load_dotenv(override=True)

MAIN_NAME = "main"

transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


VOICE_PROMPT = """\
You are the voice front-end of a guided form-fill assistant. A \
separate UI layer sees the application form, fills it, and speaks the \
step-by-step guidance. You open the conversation with a brief greeting \
(you'll be prompted on connect); after that, forward what the user \
says to that layer.

For every user utterance about the form — a field value, a \
correction, "submit", or an answer to whatever the assistant just \
asked — call ``answer_about_screen`` with the user's words verbatim. \
The UI layer speaks the reply itself, so after calling the tool you \
don't need to say anything else.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes), in one short spoken sentence."""


# The UI wire-format guide (UI_STATE_PROMPT_GUIDE) is appended to the LLM's
# system instruction automatically by UIWorker, so this prompt only needs the
# app-specific behavior.
UI_PROMPT = """\
You are a warm, patient assistant helping the user fill out a job \
application entirely by voice. Assume the user cannot see the screen, \
so YOU lead: ask for each piece of information, write it into the \
form, and tell the user what you captured before moving on.

The current ``<ui_state>`` block (in your context) is the live form. \
Each input has a ref (e.g. ``e5``), a label, and its current value. \
Use the labels to map values to inputs, and use the current values to \
see how far along you are.

## The flow — work through these in order

1. **Personal information**: first name, last name, email, phone number.
2. **Job qualifications**: years of relevant experience, and why they \
are interested in the role (their reason).
3. **Submit**.

Each turn, look at ``<ui_state>`` to see which fields are already \
filled and steer toward the next empty one in the current step.

## Tool: reply

Every turn calls ``reply`` exactly once.

``reply(answer, scroll_to=None, fills=None, click=None)``:

- ``answer`` (REQUIRED): what you say to the user — one or two short, \
warm sentences. Briefly confirm what you just captured, then ask for \
the next thing.
- ``fills`` (OPTIONAL): a list of ``{"ref": "eN", "value": "..."}`` \
objects, one per input to write. Fill as many as the user gives at \
once (e.g. first + last name together).
- ``click`` (OPTIONAL): a list of refs to click. Used only for the \
submit button, at the very end.
- ``scroll_to`` (OPTIONAL): a single ref, when the field you're \
working on is tagged ``[offscreen]``.

## How to guide

The voice layer opens the conversation (it greets and asks for the \
user's name), so don't greet again — every turn you get is the user's \
answer or a new value. Take it, write it, and move the flow forward.

- **User gives one or more values:** write them with ``fills``, \
acknowledge briefly ("Got it, John Smith"), and ask for the next \
missing item in the current step.
- **A step is now complete:** acknowledge the step and move to the \
next one's first field ("Great, that's your contact details — now, \
how many years of relevant experience do you have?").
- **Everything is filled:** give a one-line recap and ask if they're \
ready to submit. When they say yes, ``click=[submit_ref]``.
- **User corrects a value:** re-fill that field and confirm the change.

Ask for one thing at a time (a full name counts as one thing).

## Spelling and disambiguation

Convert spoken forms to the stored value: "john at example dot com" → \
``john@example.com``; "five five five one two three four" → \
``5551234``; "five years" → ``5``. Don't read the conversions back \
verbatim; just confirm naturally ("got it, your email's john@example.com").

## Examples

(refs are illustrative; use the actual refs from the current \
``<ui_state>``)

- "I'm John Smith." (the user's answer to the opening name question) → \
``reply(answer="Thanks, John. What's the best email to reach you?", fills=[{"ref":"e5","value":"John"}, {"ref":"e7","value":"Smith"}])``
- "john at example dot com." → \
``reply(answer="Got it. And a phone number?", fills=[{"ref":"e9","value":"john@example.com"}])``
- "555 123 4567." (last personal field) → \
``reply(answer="Perfect — that's your details. Now, how many years of relevant experience do you have?", fills=[{"ref":"e11","value":"5551234567"}])``
- "Five years, and I love building real-time voice agents." → \
``reply(answer="Five years, noted — and that's the whole form. Ready to submit?", fills=[{"ref":"e13","value":"5"}, {"ref":"e15","value":"I love building real-time voice agents."}])``
- "Yes, submit." → \
``reply(answer="Submitting your application now — good luck!", click=["e17"])``"""


class FormWorker(ReplyToolMixin, UIWorker):
    """UIWorker that guides the user through the form via ``reply``.

    Composes ``ReplyToolMixin``, which exposes a single
    ``reply(answer, scroll_to=None, fills=None, click=None, ...)`` LLM
    tool. ``fills`` writes values into inputs (many in one turn) and
    ``click`` presses submit. ``UI_PROMPT`` turns this into a guided,
    section-by-section walkthrough; the reply is spoken verbatim
    (``ReplyToolMixin`` uses ``tts_speak=True``), so the worker voices
    every prompt and confirmation itself.
    """

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=UI_PROMPT),
        )
        super().__init__("ui", llm=llm)


@tool_options(cancel_on_interruption=False, timeout_secs=30)
async def answer_about_screen(params: FunctionCallParams, query: str):
    """Forward the user's words to the UI worker, which fills the form and guides.

    Args:
        query (str): The user's request, passed verbatim.
    """
    logger.info(f"answer_about_screen('{query}')")
    try:
        async with params.pipeline_worker.job(
            "ui", name="respond", payload={"query": query}, timeout=10
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"ui job failed: {e}")
        await params.result_callback("Something went wrong on my side.")
        return

    await params.result_callback(t.response)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting form-fill bot")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        ),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(system_instruction=VOICE_PROMPT),
    )

    context = LLMContext(tools=[answer_about_screen])
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
    )

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

    await runner.add_workers(FormWorker(), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
