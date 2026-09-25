#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deixis: the voice LLM asks the UI worker what the user selected, and points back.

The page renders an article. The user selects a paragraph and asks
"explain this" or "rephrase that". The voice LLM cannot see the page, so
it calls ``selection()``, which returns the selected text, and answers
from it. For "where does it talk about RNA editing?" it calls
``screen("select_text", "the paragraph about RNA editing")``: the UI
worker's classifier picks that paragraph and the page selects it, so the
user sees exactly what the bot means.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        ├── selection() tool          → job "selection" on the UI worker
        └── screen(action, target)    → job "screen" on the UI worker

    DeixisWorker (UIWorker with a classifier, no LLM turn):
      ├── @job("selection"): the selected text, read from the snapshot
      └── built-in "screen" job: select_text / scroll_to / highlight by description

Run::

    uv run bot.py

Then open the client at ``http://localhost:5173`` (see ``README.md``).

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- TYPESAFE_API_KEY (optional; uses Jev as the classifier)
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus.messages import BusJobRequestMessage
from pipecat.classifiers.jev.classifier import JevClassifier
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.job_context import JobError, JobParams
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
You help the user read an article on their screen. You cannot see the \
page and you cannot know what the user has selected; only your tools \
can. Whenever the user says "this", "that" or "this paragraph", call a \
tool first and go by what it returns. Never say nothing is selected on \
your own.

## Tools

- selection(): the text the user has selected. Call it for "explain \
this", "rephrase that", "what does this mean" and any other question \
about the selection, then answer from the text it returns.
- screen(action, target): "select_text" with a description such as \
"the paragraph about RNA editing" for "where does it talk about ..." \
or "show me the part about ...". The page selects that paragraph and \
scrolls to it, so just say where it is, such as "Here, in the \
paragraph about RNA editing." "highlight" flashes an element briefly \
for short emphasis.

Keep replies to one or two short spoken sentences. No markdown, no \
lists, no symbols."""


class DeixisWorker(UIWorker):
    """UIWorker that answers the selection and screen jobs, with a classifier and no LLM turn."""

    def __init__(self):
        llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"])
        api_key = os.getenv("TYPESAFE_API_KEY")
        classifier = JevClassifier(api_key=api_key) if api_key else None
        super().__init__(UI_NAME, llm=llm, classifier=classifier)

    @job(name="selection")
    async def _selection_job(self, message: BusJobRequestMessage) -> None:
        selection = (self._latest_snapshot or {}).get("selection")
        text = selection.get("text") if isinstance(selection, dict) else None
        text = text.strip() if isinstance(text, str) else ""
        logger.debug(f"{self}: selection is {text[:60]!r}" if text else f"{self}: no selection")
        await self.send_job_response(message.job_id, {"text": text or None})


@tool_options(cancel_on_interruption=False, timeout_secs=10)
async def selection(params: FunctionCallParams):
    """The text the user has selected on the page.

    Call it whenever the user refers to "this", "this paragraph" or a
    selection; nothing else can tell whether anything is selected. Returns
    the selected text, or no text when nothing is selected.

    Args:
        params: Framework-provided tool invocation context.
    """
    try:
        async with params.pipeline_worker.job(
            UI_NAME, params=JobParams(name="selection", timeout=10)
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"ui job selection failed: {e}")
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback(t.response)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting deixis bot")

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

    context = LLMContext(tools=[selection, *screen_tools(UI_NAME)])
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

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(DeixisWorker(), worker)

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

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
