#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hello UIWorker — the smallest possible accessibility-snapshot demo.

A voice bot whose LLM delegates every screen-relevant utterance to a
``UIWorker`` that sees the page and writes the spoken answer.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── answer_about_screen(query) tool
              └── params.pipeline_worker.job("hello", name="respond", payload={query})

    HelloWorker (UIWorker):
      └── @tool answer(text)

The main LLM is the conversational layer: it forwards every utterance
to the UI worker via the ``answer_about_screen`` tool and speaks the
result. The UI worker's built-in ``respond`` job fires, which
auto-injects the latest ``<ui_state>`` block into its LLM context. The
UI worker's LLM picks the ``answer`` tool with a spoken reply grounded
in what's on screen.

The RTVI⇄bus UI wiring is built into ``PipelineWorker`` (active because
``enable_rtvi=True``), so inbound ``ui-snapshot`` messages from the
client are broadcast on the bus and the ``UIWorker`` stores them — no
decorator or manual wiring needed.

Why two LLMs for "hello world": this is the pattern UIWorker's
auto-inject is built for. The UI worker auto-injects the current screen
at the start of every delegated job, so the conversational LLM stays
small and screen-unaware. Later examples (deixis, form-fill,
async-tasks) compose new tools onto the same skeleton.

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
from pipecat.workers.llm import tool
from pipecat.workers.runner import WorkerRunner
from pipecat.workers.ui import UIWorker

load_dotenv(override=True)

MAIN_NAME = "main"

transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


VOICE_PROMPT = """\
You are the voice layer of a screen-aware assistant. A separate UI \
layer sees the page the user is looking at and writes the spoken \
reply for any question that could plausibly involve the page.

## Routing rule
For every user utterance that could involve the page in any way — \
"what's on screen", "what does this say", "is X on the page", \
factual questions, navigational questions, anything where the page \
content might matter — call ``answer_about_screen`` with the user's \
request verbatim. The tool's response is the spoken reply, already \
TTS-ready; pass it through without paraphrasing.

If the request has nothing to do with the page, still call the \
tool — the UI layer falls back to general knowledge.

## When to answer directly
Only respond directly for pure pleasantries that don't need any \
content awareness:

- Greetings ("hi", "hello").
- Acknowledgements ("thanks", "got it").
- Goodbyes ("bye", "see you").

Keep direct replies to one short spoken sentence. No markdown, no \
lists, no symbols."""


# The UI wire-format guide (UI_STATE_PROMPT_GUIDE) is appended to the LLM's
# system instruction automatically by UIWorker, so this prompt only needs the
# app-specific behavior.
HELLO_PROMPT = """\
You answer the user's question grounded in the page they're looking \
at. The current ``<ui_state>`` block is in your context — use it for \
anything the user could be asking about on screen.

Always call exactly one tool: ``answer(text)``. Put the spoken reply \
in ``text``. Plain language, one or two short sentences, no markdown \
or symbols.

When the question is about something on the page, ground claims in \
the ``<ui_state>`` content. When it's general knowledge with no \
on-page referent (history, geography, definitions), answer from your \
own knowledge. Don't tell the user what you can't see — just answer \
or admit you don't know."""


class HelloWorker(UIWorker):
    """Snapshot-aware layer. Answers grounded in ``<ui_state>``.

    A ``UIWorker`` is an always-on delegate: it comes online to receive
    snapshots and ``respond`` jobs as soon as its pipeline starts.
    """

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=HELLO_PROMPT),
        )
        super().__init__("hello", llm=llm)

    @tool
    async def answer(self, params: FunctionCallParams, text: str):
        """Speak ``text`` back to the user.

        Args:
            text: The spoken reply in plain language. One or two short
                sentences. No markdown, no symbols, no lists.
        """
        logger.info(f"{self}: answer('{text[:80]}…')")
        await self.respond_to_job(text, tts_speak=True)
        await params.result_callback(None)


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def answer_about_screen(params: FunctionCallParams, query: str):
    """Ask the screen-aware UI layer to answer about the current page.

    Args:
        query (str): The user's request, passed verbatim.
    """
    logger.info(f"answer_about_screen('{query}')")
    try:
        async with params.pipeline_worker.job(
            "hello", name="respond", payload={"query": query}, timeout=30
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"hello job failed: {e}")
        await params.result_callback("Something went wrong on my side.")
        return

    await params.result_callback(t.response)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting hello-snapshot bot")

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
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user briefly. Tell them they can ask about "
                    "anything on this page. One short sentence."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(HelloWorker(), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
