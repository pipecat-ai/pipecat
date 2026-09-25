#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async tasks: fan out long-running work and stream progress to the client.

The user asks the assistant to research a topic. The voice LLM calls the
``research`` tool, which sends a job to the UI worker; the worker
dispatches three peer workers (Wikipedia, news, scholarly papers) in
parallel as a job group, and every group a ``UIWorker`` dispatches is
reported to the client as it goes. Each peer emits progress while it
works; the client draws an in-flight card with per-worker status, and the
user can cancel the group mid-flight from the card.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in -> STT -> user_agg -> LLM -> TTS -> transport.out -> assistant_agg
        └── research(query) tool
              └── params.pipeline_worker.job("ui", name="research", payload={query})

    ResearchWorker (UIWorker "ui"):
      ├── @job research    -> request_job_group("wikipedia", "news", "scholar",
      │                          params=JobGroupParams(payload=..., label=...))
      └── on_job_completed -> say("The research on ... is done.")

    Three peer workers (BaseWorker each):
      WikipediaResearcher · NewsResearcher · ScholarResearcher

The workers are deliberately simulated with ``asyncio.sleep`` and canned
summaries so the demo focuses on the protocol, not the AI. A real app
would wire each worker to its own data source.

``request_job_group`` dispatches the group fire-and-forget and returns
at once, so the spoken "researching X" acknowledgement frees the voice
LLM to take new turns while the workers continue. Results land on the
page as they arrive.

Run::

    uv run bot.py

Then open the client at ``http://localhost:5173`` (see ``README.md``).
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
from pipecat.pipeline.job_context import JobError, JobGroupParams, JobGroupResponse
from pipecat.pipeline.job_decorator import job
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
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.runner import WorkerRunner
from pipecat.workers.ui import UIWorker

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
    try:
        async with params.pipeline_worker.job(
            UI_NAME, name="research", payload={"query": query}, timeout=10
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"research job failed: {e}")
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback(
        {
            "status": "started",
            "job_id": t.response["job_id"],
            "note": "Workers run in the background; results stream to the user's screen.",
        }
    )


class ResearchWorker(UIWorker):
    """UIWorker that fans research out to the peer workers as a client-visible group.

    The cards on the client show the progress; when every worker has
    finished, the worker also says so, since the user may not be looking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queries: dict[str, str] = {}

    @job(name="research")
    async def _research(self, message: BusJobRequestMessage) -> None:
        query = str((message.payload or {}).get("query", ""))
        job_id = await self.request_job_group(
            "wikipedia",
            "news",
            "scholar",
            params=JobGroupParams(payload={"query": query}, label=f"Research: {query}"),
        )
        self._queries[job_id] = query
        await self.send_job_response(message.job_id, {"job_id": job_id})

    async def on_job_completed(self, result: JobGroupResponse) -> None:
        await super().on_job_completed(result)
        query = self._queries.pop(result.job_id, None)
        if query:
            await self.say(f"The research on {query} is done. The results are on your screen.")

    async def cancel_job_group(self, job_id: str, *, reason: str | None = None) -> None:
        self._queries.pop(job_id, None)
        await super().cancel_job_group(job_id, reason=reason)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting async-tasks bot")

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

    # The UI worker dispatches the client-visible job groups; its card
    # messages reach the client through the main worker's RTVI bridge.
    ui = ResearchWorker(UI_NAME, llm=OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"]))

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(
        ui,
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
