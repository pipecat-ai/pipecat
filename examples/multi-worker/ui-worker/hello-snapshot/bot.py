#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hello UIWorker: the smallest example of a voice LLM asking a UIWorker about the page.

The voice LLM cannot see the page. For any question that could be about
it, it calls ``ask_page(question)``, which sends the UI worker's built-in
``respond`` job. The UI worker's LLM sees the latest accessibility
snapshot of the page, answers in a sentence or two through its
``answer`` tool, and that answer comes back to the voice LLM, which
speaks it.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── ask_page(question) tool → job "respond" on the UI worker

    HelloWorker (UIWorker):
      └── @tool answer(text) → the job's response

``PipelineWorker`` connects the UI worker to the client on its own (RTVI
is enabled by default): the client streams snapshots of the page and the
worker keeps the latest one.

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
from pipecat.pipeline.job_context import JobError, JobParams
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
from pipecat.workers.llm import tool
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
You are a voice assistant. You cannot see the page the user is looking \
at; the ``ask_page`` tool can. For any question that could be about the \
page, such as "what's on screen", "what does the second story say" or \
"is X on the page", call ``ask_page`` with the user's question and \
answer from what it returns. Answer greetings, thanks and goodbyes \
yourself.

Keep replies to one or two short spoken sentences. No markdown, no \
lists, no symbols."""


HELLO_PROMPT = """\
You answer questions about the page the user is looking at. Reply with \
the ``answer`` tool, in one or two plain sentences. When the question \
is not about the page, answer from general knowledge. Don't tell the \
user what you can't see; answer, or say you don't know."""


class HelloWorker(UIWorker):
    """UIWorker whose LLM answers questions about the page for the voice LLM."""

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=HELLO_PROMPT),
        )
        super().__init__(UI_NAME, llm=llm)

    @tool
    async def answer(self, params: FunctionCallParams, text: str):
        """Answer the question with ``text``.

        Args:
            text: The answer in plain language. One or two short sentences.
                No markdown, no symbols, no lists.
        """
        logger.debug(f"{self}: answer({text[:80]!r})")
        await self.respond_to_job(text)
        await params.result_callback(None)


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def ask_page(params: FunctionCallParams, question: str):
    """Ask about the page the user is looking at.

    Call it for any question that could be about the page; nothing else
    can see it. Returns the answer.

    Args:
        params: Framework-provided tool invocation context.
        question: The user's question, as they asked it.
    """
    try:
        async with params.pipeline_worker.job(
            UI_NAME, params=JobParams(name="respond", payload={"query": question}, timeout=30)
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"ui job respond failed: {e}")
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback(t.response)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting hello-snapshot bot")

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

    context = LLMContext(tools=[ask_page])
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
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(HelloWorker(), worker)

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

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
