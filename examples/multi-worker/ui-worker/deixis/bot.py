#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deixis — the UIWorker grounds in what the user just selected.

The page renders an article. The user selects a paragraph (or any span
of text) and asks "explain this", "rephrase that", "where does it talk
about RNA editing?", and so on. The client captures
``window.getSelection()`` and emits a ``<selection ref="...">selected
text</selection>`` block in the snapshot. The UIWorker reads it as a
deictic reference: "this paragraph" resolves to the selected element.

Two directions:

- **Read**: user selects text → ``<selection>`` block in ``<ui_state>``
  → the worker grounds its answer in the selected content.
- **Write**: the worker says "this paragraph" → ``select_text=ref`` puts
  the page's text selection on that element → the user sees what the
  worker is referring to.

``DeixisWorker`` composes ``ReplyToolMixin``: the ``reply(answer,
scroll_to, highlight, select_text)`` bundle covers attention-pointing
(scroll / highlight) and reading-style (selection) apps.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── answer_about_screen(query) tool
              └── params.pipeline_worker.job("ui", name="respond", payload={query})

    DeixisWorker (ReplyToolMixin + UIWorker):
      └── inherited: reply(answer, scroll_to, highlight, select_text)

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
You are the voice layer of a screen-aware reading assistant. A \
separate UI layer sees the page (and the user's selection) and \
writes the spoken reply.

For every user utterance about the article, call \
``answer_about_screen`` with the user's request verbatim. The tool's \
response is the spoken reply, already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


# The UI wire-format guide (UI_STATE_PROMPT_GUIDE) is appended to the LLM's
# system instruction automatically by UIWorker, so this prompt only needs the
# app-specific behavior.
UI_PROMPT = """\
You help the user read and understand an article. The current \
``<ui_state>`` block is in your context, and may contain a \
``<selection>`` block when the user has highlighted text.

## Tool: reply

Every turn calls ``reply`` exactly once. One tool call per turn, no \
chaining.

``reply(answer, scroll_to=None, highlight=None, select_text=None)``:

- ``answer`` (REQUIRED): the spoken reply, plain language, two short \
sentences max. No markdown, no symbols, no quoting long passages.
- ``scroll_to`` (OPTIONAL): a snapshot ref. Set when the paragraph \
you want to point at is tagged ``[offscreen]``.
- ``highlight`` (OPTIONAL): a list of snapshot refs to flash briefly. \
Use for short emphasis: "look at this fact". Don't use it for a \
whole paragraph; ``select_text`` is better for that.
- ``select_text`` (OPTIONAL): a single snapshot ref. Sets the page's \
text selection to that element. Use this when you say "this \
paragraph" or "the section that talks about X" so the user sees \
exactly what you're referring to.

## Reading the user's selection

If ``<ui_state>`` contains a ``<selection ref="...">selected \
text</selection>`` block, the user has highlighted something. Treat \
that selection as the deictic referent for words like "this", \
"that", "this paragraph", "what I selected". Ground your answer in \
the selected content, not the article as a whole.

When answering about the user's selection, do NOT also call \
``select_text`` — they already selected it; pointing back at the \
same span is redundant.

## Decision rules

- User has a selection AND asks something deictic ("explain this", \
"rephrase that", "what does this mean") → ground in the selection. \
Just ``answer``; no visual fields.
- User asks "where does it say X?" or "show me the part about X" → \
find the matching paragraph, ``answer`` briefly, set \
``select_text=ref`` to point at it, and ``scroll_to=ref`` if it's \
``[offscreen]``.
- User asks a content question without selection → ``answer`` with \
the relevant fact. Optionally set ``select_text=ref`` if the \
answer is sourced from one specific paragraph.

## Examples

(refs are illustrative; use the actual refs from the current \
``<ui_state>``)

- User selects the third paragraph, asks "explain this" → \
``reply(answer="The skin acts as its own light sensor. Even though \
octopuses are colorblind, their skin can detect light directly, \
which is how they match colors so accurately.")``
- "Where does it talk about RNA editing?" (paragraph e15, offscreen) \
→ ``reply(answer="Here, in the paragraph about RNA editing.", \
scroll_to="e15", select_text="e15")``
- "How many neurons does an octopus have?" (no selection) → \
``reply(answer="About five hundred million, with two thirds of \
them in the arms.", select_text="e7")``
- "Hi, what's this article about?" (no selection) → \
``reply(answer="It's a short essay on octopus cognition. Select any \
paragraph and I'll explain it.")``"""


class DeixisWorker(ReplyToolMixin, UIWorker):
    """UIWorker that grounds in the user's selection and points back via select_text.

    Composes ``ReplyToolMixin``, which exposes a single
    ``reply(answer, scroll_to=None, highlight=None, select_text=None, ...)``
    LLM tool. The same bundle pointing apps use also covers
    reading-style apps: ``select_text`` is for "this paragraph" / "the
    section about X" (durable text selection), while ``highlight``
    flashes briefly for short emphasis.
    """

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=UI_PROMPT),
        )
        super().__init__("ui", llm=llm)


@tool_options(cancel_on_interruption=False, timeout_secs=30)
async def answer_about_screen(params: FunctionCallParams, query: str):
    """Ask the screen-aware UI worker to answer about the article / selection.

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
    logger.info("Starting deixis bot")

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
                    "Greet the user briefly. Tell them they can select a "
                    "paragraph and ask you to explain or rephrase it. One "
                    "short sentence."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(DeixisWorker(), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
