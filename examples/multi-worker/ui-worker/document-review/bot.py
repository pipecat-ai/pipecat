#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Document review: the synthesis demo.

A voice-driven workspace where the user reviews a draft article. The
voice LLM leads; the UI worker grounds words on the page with its
classifier and acts, with no LLM turn of its own. The user can:

- Select a paragraph and ask for a review. The worker reads the selection
  from its snapshot and runs two peer reviewers (clarity, tone) in
  parallel as a client-visible job group. Their progress streams to an
  in-flight card, each reviewer's feedback becomes a note attached to the
  paragraph as it lands, and the voice tells the user what they said.
- Dictate a note. The worker finds the notes textarea and the Save button
  and fills and clicks them.
- Ask "where does it talk about X". The voice LLM uses the ``screen`` tool
  to select the paragraph the classifier picks by its text.
- Ask "explain this". The voice LLM fetches the selected text and answers.
- Click an existing note; the client emits a ``note_click`` UI event and
  the worker's ``@ui_event("note_click")`` handler jumps to the paragraph.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in -> STT -> user_agg -> LLM -> TTS -> transport.out -> assistant_agg
        ├── review_selection() / add_note(text) / selection()
        └── screen(action, target, value)
              └── params.pipeline_worker.job("ui", name=..., payload=...)

    ReviewWorker (UIWorker "ui", no LLM turn):
      ├── @job review     -> job_group("clarity", "tone", ...) on the selection,
      │                      answers with both reviewers' feedback
      ├── @job add_note   -> classifier finds the textarea and Save, fills and clicks
      ├── @job selection  -> the selected text
      ├── @job screen     -> built in: find, select_text, scroll_to, ...
      ├── on_job_response -> add_note command for each reviewer that completes
      └── @ui_event("note_click") -> scroll_to + select_text(ref)

    Two peer workers (BaseWorker each):
      ClarityReviewer · ToneReviewer

The reviewers are simulated, like async-tasks: a few ``send_job_update``
progress lines, then a ``send_job_response`` with a final analysis
computed from simple text metrics (word/sentence counts, absolutist /
hedging words) so different paragraphs get different feedback without
real NLP.

With ``TYPESAFE_API_KEY`` set the worker uses Jev; otherwise its own LLM
answers through an ``LLMClassifier``.

Run::

    uv run bot.py

Then open the client at ``http://localhost:5173`` (see ``README.md``).

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- TYPESAFE_API_KEY (optional, for Jev)
"""

import asyncio
import os
import random

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus.messages import BusJobRequestMessage, BusJobResponseMessage
from pipecat.classifiers.jev.classifier import JevClassifier
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.job_context import JobError, JobGroupError, JobGroupParams, JobStatus
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
from pipecat.workers.ui import UIWorker, ui_event
from pipecat.workers.ui.ui_tools import screen_tools

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
You are a document review assistant. The user is reading a draft \
article with a notes panel beside it. You cannot see the page and you \
cannot know what the user has selected; only your tools can. Whenever \
the user says "this", "this paragraph" or "that", call a tool first \
and go by what it returns. Never say nothing is selected on your own.

## Tools

- review_selection(): review the paragraph the user has selected with \
two reviewers. Say "Reviewing this paragraph" in the same turn as the \
call; it returns their feedback a few seconds later, which you then \
give in one or two spoken sentences.
- add_note(text): add a note to the notes panel, attached to the \
selected paragraph. Pass the note's text as it should read; resolve \
"that" from the conversation, never pass the pronoun.
- selection(): the text the user has selected. Call it for "explain \
this", "rephrase that", "what does this mean" and any other question \
about the selection, then answer from the text it returns.
- screen(action, target): "select_text" or "scroll_to" with a \
description such as "the paragraph about circadian rhythms" for \
"where does it talk about ...". "find" to check what a description \
refers to.

## Rules

- Only when a tool has answered that nothing is selected, ask the user \
to select a paragraph first.
- Answer pleasantries directly, in one short sentence.
- Your replies are spoken aloud: plain language, one or two short \
sentences, no markdown or symbols."""


class _SimulatedReviewer(BaseWorker):
    """Base for the two simulated reviewers."""

    source_name: str = "reviewer"

    def review(self, text: str) -> str:
        return ""

    async def on_job_request(self, message: BusJobRequestMessage) -> None:
        await super().on_job_request(message)
        job_id = message.job_id
        text = str((message.payload or {}).get("text", "")).strip()
        try:
            await asyncio.sleep(random.uniform(0.4, 0.9))
            await self.send_job_update(job_id, {"text": f"reading {len(text.split())} words"})

            await asyncio.sleep(random.uniform(0.5, 1.1))
            await self.send_job_update(job_id, {"text": f"checking {self.source_name}"})

            await asyncio.sleep(random.uniform(0.4, 0.9))
            feedback = self.review(text) or "(no notes)"
            await self.send_job_response(job_id, response={"feedback": feedback})
        except asyncio.CancelledError:
            raise


class ClarityReviewer(_SimulatedReviewer):
    """Comments on density, sentence length, and structural issues."""

    source_name = "clarity"

    def review(self, text: str) -> str:
        words = len(text.split())
        # Cheap sentence count: terminal punctuation.
        sentences = max(1, sum(1 for ch in text if ch in ".!?"))
        avg = words / sentences

        if avg > 35:
            return (
                f"This passage runs {words} words across just {sentences} "
                f"sentence(s) (~{avg:.0f} words each). Consider breaking "
                "it into smaller units; the reader is asked to hold a lot "
                "in working memory."
            )
        if words < 25:
            return (
                f"Brief at {words} words. If this is a key idea, consider "
                "expanding with one concrete example."
            )
        if avg < 12:
            return (
                f"Sentences average {avg:.0f} words. This is fine, "
                "sometimes preferable, but watch for choppiness if "
                "several short ones run in a row."
            )
        return (
            f"Density is reasonable at ~{avg:.0f} words per sentence across {sentences} sentences."
        )


class ToneReviewer(_SimulatedReviewer):
    """Comments on hedging, overstatement, and word choice."""

    source_name = "tone"

    ABSOLUTIST = (
        "simply",
        "anyone who",
        "unanimous",
        "always",
        "never",
        "obviously",
        "comprehensively",
    )
    HEDGES = ("might", "perhaps", "seems", "appears", "could", "may")

    def review(self, text: str) -> str:
        lower = text.lower()
        absolutes = [w for w in self.ABSOLUTIST if w in lower]
        hedges = [w for w in self.HEDGES if w in lower]

        if absolutes:
            sample = ", ".join(repr(w) for w in absolutes[:3])
            return (
                f"Strong words flagged: {sample}. If the claim is contested "
                "or the evidence is mixed, some hedging would read as more "
                "credible."
            )
        if len(hedges) >= 4:
            return (
                f"Heavy hedging — I count {len(hedges)} hedge words. Fine "
                "for an exploratory section, but if you mean to commit to "
                "a claim, the hedges weaken it."
            )
        return "Tone reads as measured. No flags."


# ─────────────────────────────────────────────────────────────────────
# Review UI worker.
# ─────────────────────────────────────────────────────────────────────


class ReviewWorker(UIWorker):
    """UIWorker that works the review page for the voice LLM, with a classifier and no LLM turn.

    Reviews run as a client-visible job group; each reviewer's response
    becomes a note on the page as it lands, through ``on_job_response``,
    and the job answers with all the feedback once the group is done.
    """

    def __init__(self):
        llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"])
        api_key = os.getenv("TYPESAFE_API_KEY")
        classifier = JevClassifier(api_key=api_key) if api_key else None
        super().__init__(UI_NAME, llm=llm, classifier=classifier)
        # job_id -> the paragraph under review, so on_job_response can
        # attach each reviewer's feedback to the right note.
        self._reviews: dict[str, str] = {}

    @job(name="review")
    async def _review(self, message: BusJobRequestMessage) -> None:
        selected = self._selection()
        if not selected:
            await self.send_job_response(
                message.job_id, {"error": "nothing is selected"}, status=JobStatus.ERROR
            )
            return
        ref, text = selected
        job_id: str | None = None
        try:
            async with self.job_group(
                "clarity",
                "tone",
                params=JobGroupParams(
                    payload={"ref": ref, "text": text}, label=f"Reviewing ¶ {ref}", timeout=30
                ),
            ) as group:
                job_id = group.job_id
                self._reviews[job_id] = ref
        except JobGroupError as e:
            logger.warning(f"{self}: review of {ref!r} failed: {e}")
            await self.send_job_response(message.job_id, {"error": str(e)}, status=JobStatus.ERROR)
            return
        finally:
            if job_id:
                self._reviews.pop(job_id, None)
        feedback = {
            name: (response or {}).get("feedback") for name, response in group.responses.items()
        }
        await self.send_job_response(message.job_id, {"feedback": feedback})

    @job(name="add_note")
    async def _add_note(self, message: BusJobRequestMessage) -> None:
        text = str((message.payload or {}).get("text", "")).strip()
        textarea = await self.which_element("the notes textarea")
        save = await self.which_element("the Save button")
        if not text or not textarea or not save:
            await self.send_job_response(message.job_id, {"done": False})
            return
        await self.set_input_value(textarea, text)
        await self.click(save)
        await self.send_job_response(message.job_id, {"done": True})

    @job(name="selection")
    async def _selection_job(self, message: BusJobRequestMessage) -> None:
        selected = self._selection()
        await self.send_job_response(message.job_id, {"text": selected[1] if selected else None})

    async def on_job_response(self, message: BusJobResponseMessage) -> None:
        """Turn reviewer responses into ``add_note`` UI commands."""
        await super().on_job_response(message)
        ref = self._reviews.get(message.job_id)
        if not ref or message.status != JobStatus.COMPLETED:
            return
        feedback = ((message.response or {}).get("feedback") or "").strip()
        if not feedback:
            return
        await self.send_command(
            "add_note", {"source": message.source, "ref": ref, "text": feedback}
        )

    @ui_event("note_click")
    async def on_note_click(self, message) -> None:
        """User clicked a note in the panel; jump to its paragraph."""
        ref = (message.payload or {}).get("ref")
        if not isinstance(ref, str) or not ref:
            return
        logger.info(f"{self}: note_click -> scroll_to + select_text({ref!r})")
        await self.scroll_to(ref)
        await self.select_text(ref)

    def _selection(self) -> tuple[str, str] | None:
        """The user's current selection from the snapshot, as (ref, text), or None."""
        snapshot = self._latest_snapshot or {}
        selection = snapshot.get("selection")
        if not isinstance(selection, dict):
            logger.debug(
                f"{self}: no selection in the snapshot "
                f"(captured_at={snapshot.get('captured_at')}, keys={sorted(snapshot)})"
            )
            return None
        ref, text = selection.get("ref"), selection.get("text")
        if not isinstance(ref, str) or not ref or not isinstance(text, str) or not text.strip():
            logger.debug(f"{self}: unusable selection in the snapshot: {selection!r}")
            return None
        return ref, text.strip()


async def _ui(params: FunctionCallParams, name: str, payload: dict, timeout: float = 30) -> None:
    """Send a job to the UI worker and hand its answer to the voice LLM."""
    try:
        async with params.pipeline_worker.job(
            UI_NAME, name=name, payload=payload, timeout=timeout
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"ui job {name} failed: {e}")
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback(t.response)


@tool_options(cancel_on_interruption=False, timeout_secs=45)
async def review_selection(params: FunctionCallParams):
    """Review the paragraph the user has selected with two reviewers, clarity and tone.

    Takes a few seconds; the user sees their progress on screen. Returns
    each reviewer's feedback, or an error when nothing is selected.

    Args:
        params: Framework-provided tool invocation context.
    """
    await _ui(params, "review", {}, timeout=45)


@tool_options(cancel_on_interruption=False, timeout_secs=15)
async def add_note(params: FunctionCallParams, text: str):
    """Add a note to the notes panel, attached to the paragraph the user selected.

    Args:
        params: Framework-provided tool invocation context.
        text: The note, as it should read.
    """
    await _ui(params, "add_note", {"text": text}, timeout=15)


@tool_options(cancel_on_interruption=False, timeout_secs=10)
async def selection(params: FunctionCallParams):
    """The text the user has selected on the page.

    Call it whenever the user refers to "this", "this paragraph" or a
    selection; nothing else can tell whether anything is selected. Returns
    the selected text, or no text when nothing is selected.

    Args:
        params: Framework-provided tool invocation context.
    """
    await _ui(params, "selection", {}, timeout=10)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting document-review bot")

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

    context = LLMContext(tools=[review_selection, add_note, selection, *screen_tools(UI_NAME)])
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

    await runner.add_workers(
        ReviewWorker(),
        ClarityReviewer("clarity"),
        ToneReviewer("tone"),
        worker,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user briefly. Tell them they can select any "
                    "paragraph and ask you to review it, dictate notes, or "
                    "navigate the draft. One short sentence."
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
