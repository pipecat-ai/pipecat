#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async tasks — the UIWorker fans out long-running work and streams progress.

The user asks the assistant to research a topic. The UIWorker dispatches
three peer workers (Wikipedia, news, scholarly papers) in parallel via
``start_ui_job_group``. Each worker emits progress updates while it
works. ``UIWorker`` forwards every lifecycle event to the client as
``ui-job-group`` envelopes (``group_started``, ``job_update``,
``job_completed``, ``group_completed``), which the client renders as
in-flight cards with per-worker status. The user can cancel a group
mid-flight via ``client.cancelUIJobGroup(job_id)``, which sends a reserved
``__cancel_job_group`` event that the worker turns into a ``cancel_job_group``
call.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── answer_about_screen(query) tool
              └── params.pipeline_worker.job("ui", name="respond", payload={query})

    ResearchWorker (UIWorker):
      └── @tool reply(answer, research_query=None)
            └── (if research_query) start_ui_job_group("wikipedia", "news", "scholar")

    Three peer workers (BaseWorker each):
      WikipediaResearcher · NewsResearcher · ScholarResearcher

The workers are deliberately simulated with ``asyncio.sleep`` and canned
summaries so the demo focuses on the protocol, not the AI. A real app
would wire each worker to its own data source.

``start_ui_job_group`` dispatches the group on a background task and
returns immediately, so the spoken "researching X" acknowledgement frees
the main LLM to take new turns while the workers continue.

Run::

    uv run bot.py

Then open the client at ``http://localhost:5173`` (see ``README.md``).

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import asyncio
import os
import random

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus.messages import BusJobRequestMessage
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
from pipecat.workers.base_worker import BaseWorker
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
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


VOICE_PROMPT = """\
You are the voice layer of a research assistant. A separate UI \
layer sees the page and dispatches research tasks.

For every user utterance involving research (asking about a topic, \
launching a search, asking for follow-ups), call \
``answer_about_screen`` with the user's request verbatim. The \
tool's response is the spoken reply, already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


# The UI wire-format guide (UI_STATE_PROMPT_GUIDE) is appended to the LLM's
# system instruction automatically by UIWorker, so this prompt only needs the
# app-specific behavior.
UI_PROMPT = """\
You help the user research topics. When the user names something \
to look up, kick off a parallel research task across three worker \
sources (Wikipedia, news, scholarly papers).

## Tool: reply

Every turn calls ``reply`` exactly once. One tool call per turn.

``reply(answer, research_query=None)``:

- ``answer`` (REQUIRED): the spoken reply, plain language, one \
short sentence. No markdown, no symbols.
- ``research_query`` (OPTIONAL): the topic to research. When set, \
the server fans out three workers in parallel and streams \
their progress to an in-flight panel on the page. The workers run \
in the background; you do NOT wait for results. Just speak a brief \
acknowledgement.

## Decision rules

- **User asks to research / look up / find out about something** → \
set ``research_query`` to the topic and answer with a brief \
acknowledgement ("Researching the Mariana Trench now"). The server \
handles the rest; results stream onto the page.
- **User asks a quick question you can answer immediately** → just \
``answer``. Don't kick off a research task for trivia or for \
questions about the in-flight tasks themselves.
- **User asks about ongoing research** → just ``answer`` (the \
results panel on screen shows progress).

## Examples

- "Research the Mariana Trench." → \
``reply(answer="Researching the Mariana Trench now.", research_query="Mariana Trench")``
- "Look up octopus cognition." → \
``reply(answer="Looking that up.", research_query="octopus cognition")``
- "How many neurons does an octopus have?" (quick question, no \
research needed) → ``reply(answer="About five hundred million.")``
- "Hi." → ``reply(answer="Hi! What would you like to research?")``"""


class _SimulatedResearcher(BaseWorker):
    """BaseWorker peer that fakes a research task with progress updates.

    Receives a ``payload={"query": ...}``. Emits a few ``send_job_update``
    messages with progress text, then a final ``send_job_response``
    carrying a canned summary. The randomized ``asyncio.sleep`` makes the
    workers feel like they run at different paces, which shows off the
    streaming UI.

    Subclasses set ``source_name`` and provide ``summarize(query)``.
    """

    source_name: str = "researcher"

    def summarize(self, query: str) -> str:
        return f"Generic results for '{query}'."

    async def on_job_request(self, message: BusJobRequestMessage) -> None:
        await super().on_job_request(message)
        job_id = message.job_id
        query = (message.payload or {}).get("query", "")
        try:
            await asyncio.sleep(random.uniform(0.4, 1.2))
            await self.send_job_update(job_id, {"text": f"searching {self.source_name}…"})

            await asyncio.sleep(random.uniform(0.6, 1.4))
            n = random.randint(3, 8)
            await self.send_job_update(job_id, {"text": f"found {n} results"})

            await asyncio.sleep(random.uniform(0.5, 1.5))
            await self.send_job_update(job_id, {"text": "summarizing"})

            await asyncio.sleep(random.uniform(0.4, 0.9))
            await self.send_job_response(job_id, response={"summary": self.summarize(query)})
        except asyncio.CancelledError:
            # The base worker's cancellation hook auto-emits a CANCELLED
            # response; just bail.
            raise


class WikipediaResearcher(_SimulatedResearcher):
    source_name = "wikipedia"

    def summarize(self, query: str) -> str:
        return (
            f"Wikipedia overview of {query}: a one-paragraph summary covering "
            "the historical background, key facts, and major figures."
        )


class NewsResearcher(_SimulatedResearcher):
    source_name = "news"

    def summarize(self, query: str) -> str:
        return (
            f"Recent news on {query}: three headlines from the past month, "
            "a short context paragraph, and any active developments."
        )


class ScholarResearcher(_SimulatedResearcher):
    source_name = "scholar"

    def summarize(self, query: str) -> str:
        return (
            f"Scholarly take on {query}: two highly cited papers, the "
            "consensus position, and a notable debate or open question."
        )


class ResearchWorker(UIWorker):
    """UIWorker that kicks off background research job groups.

    The custom ``@tool reply`` has a ``research_query`` field. When the
    LLM sets it, the tool fires ``start_ui_job_group(...)`` against the
    three peer workers — fire-and-forget from the LLM's perspective, so
    the tool returns immediately with the spoken acknowledgement. The
    ``UIWorker`` forwards every job lifecycle event to the client as
    ``ui-job-group`` envelopes, where the client renders progress and a cancel
    button.
    """

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=UI_PROMPT),
        )
        super().__init__("ui", llm=llm)

    @tool
    async def reply(
        self,
        params: FunctionCallParams,
        answer: str,
        research_query: str | None = None,
    ):
        """Reply to the user. Optionally kick off background research.

        Always called exactly once per turn. ``answer`` is required.

        Args:
            answer: The spoken reply in plain language. One short
                sentence. For research turns, a brief acknowledgement
                like "Researching X now."
            research_query: Optional topic to research. When set, the
                server fans out three workers in parallel and
                streams progress to the page. Workers run in the
                background; the LLM does NOT wait for results.
        """
        logger.info(f"{self}: reply(answer={answer!r}, research_query={research_query!r})")
        if research_query:
            await self.start_ui_job_group(
                "wikipedia",
                "news",
                "scholar",
                payload={"query": research_query},
                label=f"Research: {research_query}",
            )
        await self.respond_to_job(answer)
        await params.result_callback(None)


@tool_options(cancel_on_interruption=False, timeout_secs=30)
async def answer_about_screen(params: FunctionCallParams, query: str):
    """Forward the user's request to the screen-aware research worker.

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
    logger.info("Starting async-tasks bot")

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
                    "Greet the user briefly. Tell them they can ask you to "
                    "research any topic. One short sentence."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(
        ResearchWorker(),
        WikipediaResearcher("wikipedia"),
        NewsResearcher("news"),
        ScholarResearcher("scholar"),
        worker,
    )

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
