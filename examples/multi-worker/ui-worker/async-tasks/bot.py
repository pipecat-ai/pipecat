#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async tasks — fan out long-running work and stream progress to the client.

The user asks the assistant to research a topic. The main pipeline's own
LLM calls the ``research`` tool, which dispatches three peer workers
(Wikipedia, news, scholarly papers) in parallel via a ``BaseUIWorker``
dispatcher registered on the runner:
``request_job_group(...)`` on a ``BaseUIWorker``, with no LLM in the
dispatch path, no ``UIWorker`` required. Each peer emits progress updates while it works; the group's
lifecycle reaches the client as ``ui-job-group`` envelopes
(``group_started``, ``job_update``, ``job_completed``,
``group_completed``), which the client renders as in-flight cards with
per-worker status. The user can cancel a group mid-flight via
``client.cancelUIJobGroup(job_id)``, which sends a reserved
``__cancel_job_group`` event that the dispatching worker turns into a
``cancel_job_group`` call.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── research(query) tool
              └── ui_jobs.request_job_group(          # found by name on the runner
                      "wikipedia", "news", "scholar",
                      params=JobGroupParams(payload=..., label=...))

    ui_jobs (BaseUIWorker): the client-visible job-group dispatcher (no LLM)

    Three peer workers (BaseWorker each):
      WikipediaResearcher · NewsResearcher · ScholarResearcher

The workers are deliberately simulated with ``asyncio.sleep`` and canned
summaries so the demo focuses on the protocol, not the AI. A real app
would wire each worker to its own data source.

``request_job_group`` dispatches the group fire-and-forget and
returns immediately, so the spoken "researching X" acknowledgement frees
the LLM to take new turns while the workers continue. Results land on
the page as they arrive. (When the LLM must also *read or drive* the
page — snapshots, deixis, UI commands — reach for ``UIWorker``; see the
document-review example.)

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
from pipecat.pipeline.job_context import JobGroupParams
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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.base_ui_worker import BaseUIWorker
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.runner import WorkerRunner

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
You are a research assistant. You can fan out background research on \
any topic; progress and results stream to a panel on the user's screen.

## Tool: research

``research(query)`` starts three background workers (Wikipedia, news, \
scholarly papers) on the topic. They run in the background; you do NOT \
wait for results — they appear on the user's screen as they land. After \
calling it, speak a one-sentence acknowledgement.

## Decision rules

- **User asks to research / look up / find out about something** → call \
``research`` with the topic, then acknowledge briefly \
("Researching the Mariana Trench now.").
- **User asks a quick question you can answer immediately** → just \
answer it. Don't start research for trivia.
- **User asks about ongoing research** → tell them progress and results \
are on their screen. Don't start a duplicate task.

Your replies are spoken aloud: plain language, one short sentence, no \
markdown or symbols."""


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


@tool_options(cancel_on_interruption=False)
async def research(params: FunctionCallParams, query: str):
    """Start background research on a topic across three worker sources.

    Dispatches the workers fire-and-forget: the group's progress and
    results stream to the client as ``ui-job-group`` envelopes, so this
    tool returns immediately and the LLM speaks a short acknowledgement.

    Args:
        query (str): The topic to research, e.g. "Mariana Trench".
    """
    logger.info(f"research('{query}')")
    ui_jobs: BaseUIWorker = params.worker_runner.get_worker("ui-jobs")
    job_id = await ui_jobs.request_job_group(
        "wikipedia",
        "news",
        "scholar",
        params=JobGroupParams(
            payload={"query": query},
            label=f"Research: {query}",
        ),
    )
    await params.result_callback(
        {
            "status": "started",
            "job_id": job_id,
            "note": "Workers run in the background; results stream to the user's screen.",
        }
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting async-tasks bot")

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

    context = LLMContext(tools=[research])
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

    # The dispatcher for client-visible job groups: a plain BaseUIWorker on
    # the bus (no LLM). Tools reach it by name through the runner; its
    # envelopes reach the client through the main worker's RTVI bridge.
    ui_jobs = BaseUIWorker("ui-jobs")

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(
        ui_jobs,
        WikipediaResearcher("wikipedia"),
        NewsResearcher("news"),
        ScholarResearcher("scholar"),
        worker,
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

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
