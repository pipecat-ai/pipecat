#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Document review — the synthesis demo.

A single workspace combining everything from the prior demos. The user
reviews a draft article. They can:

- Select a paragraph and ask for review. The UIWorker fans out to two
  peer reviewers (clarity, tone) in parallel. Their progress streams to
  an in-flight card, and each worker's feedback becomes a note attached
  to the paragraph (a custom ``add_note`` command).
- Dictate their own notes by voice. The worker fills the notes textarea
  and clicks Save (``fills`` + ``click`` via the bundled ``reply`` tool).
- Ask "where does it talk about X" and the worker uses ``select_text`` to
  navigate.
- Click an existing note; the client emits a ``note_click`` UI event, and
  the worker's ``@ui_event("note_click")`` handler jumps to the related
  paragraph — the round-trip event/command pattern.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── answer_about_screen(query) tool
              └── params.pipeline_worker.job("ui", name="respond", payload={query})

    ReviewWorker (ReplyToolMixin + UIWorker, keep_history=True):
      ├── inherited: reply(answer, scroll_to, highlight, select_text, fills, click)
      ├── @tool start_review(answer, paragraph_ref, paragraph_text)
      │     └── start_ui_job_group("clarity", "tone", ...)
      ├── @ui_event("note_click") → scroll_to + select_text(ref)
      └── on_job_response → emit add_note for each reviewer that completes

    Two peer workers (BaseWorker each):
      ClarityReviewer · ToneReviewer

The reviewers are simulated, like async-tasks: a few ``send_job_update``
progress lines, then a ``send_job_response`` with a final analysis
computed from simple text metrics (word/sentence counts, absolutist /
hedging words) so different paragraphs get different feedback without
real NLP.

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
from pipecat.bus.messages import BusJobRequestMessage, BusJobResponseMessage
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.job_context import JobError, JobStatus
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
from pipecat.workers.ui import ReplyToolMixin, UIWorker, ui_event

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
You are the voice layer of a document review assistant. A separate \
UI layer sees the page (the article and the notes panel) and writes \
the spoken reply.

For every user utterance about the document or the review (selecting \
paragraphs, asking for feedback, dictating notes, navigating), call \
``answer_about_screen`` with the user's request verbatim. The \
tool's response is the spoken reply, already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


# The UI wire-format guide (UI_STATE_PROMPT_GUIDE) is appended to the LLM's
# system instruction automatically by UIWorker, so this prompt only needs the
# app-specific behavior.
UI_PROMPT = """\
You are reviewing a draft article with the user. The current \
``<ui_state>`` block is in your context, and may contain a \
``<selection>`` block when the user has highlighted text.

## The hard rule

**Every turn MUST call exactly one tool: either ``reply`` or \
``start_review``.** Never respond with plain text. If the user \
asks something that doesn't need a visual action — including \
open questions like "how can we improve it?", "what do you think?", \
"any suggestions?" — call ``reply`` with the answer in the \
``answer`` field. The spoken response is whatever you put there. \
If you forget to call a tool, the user hears nothing and the turn \
times out.

You have two LLM tools:

## Tool: reply

For most turns. ``reply(answer, scroll_to=None, highlight=None, \
select_text=None, fills=None, click=None)``:

- ``answer`` (REQUIRED): the spoken reply, plain language, one or \
two short sentences.
- ``scroll_to`` (OPTIONAL): a snapshot ref. Scroll the element into \
view.
- ``select_text`` (OPTIONAL): a snapshot ref. Place the page's text \
selection on a paragraph (use this for "this paragraph" / "the \
section about X").
- ``highlight`` (OPTIONAL): list of refs. Brief flash. Rarely used \
here; ``select_text`` is usually better for paragraphs.
- ``fills`` (OPTIONAL): list of ``{"ref", "value"}`` objects. Fill \
the notes textarea (ref is in ``<ui_state>`` as the ``textbox``).
- ``click`` (OPTIONAL): list of refs to click. Use to click the \
Save button after filling the notes textarea.

## Tool: start_review

For "review this paragraph" / "give me feedback on this" requests. \
``start_review(answer, paragraph_ref, paragraph_text)``:

- ``answer`` (REQUIRED): brief acknowledgement spoken right away \
("Reviewing this paragraph").
- ``paragraph_ref`` (REQUIRED): the snapshot ref of the paragraph \
under review. When the user has a selection, use the selection's \
ref. Otherwise pick the right paragraph from ``<ui_state>``.
- ``paragraph_text`` (REQUIRED): the full paragraph text. Read it \
from the ``<selection>`` block when present, or from the ``name`` \
attribute on the paragraph node in ``<ui_state>``.

The server fans out two worker reviewers (clarity, tone) in \
parallel and streams progress to the page. As each worker finishes, \
their feedback becomes a note attached to the paragraph. You do NOT \
wait for results.

## Decision rules

- **"Review this", "give me feedback on this paragraph", "what do \
you think of this"** with a selection → ``start_review``.
- **"Review the third paragraph"** with no selection → use \
``<ui_state>`` to find the ref + text, call ``start_review``.
- **"Add a note: …"** or any dictated note content → use ``reply`` \
with ``fills`` for the notes textarea and ``click`` on the Save \
button. The note will automatically attach to whichever article \
paragraph the user last selected.
- **"Where does it talk about X"** → ``reply`` with ``scroll_to`` + \
``select_text`` to navigate to the matching paragraph.
- **"Read me back the notes"** / **"What did you say about \
paragraph 3"** → ``reply`` with answer text only; the notes panel \
is in ``<ui_state>`` so you can summarize from it.
- **General questions about the draft** ("how can we improve it?", \
"what do you think?", "any suggestions?", "what's missing?") → \
``reply`` with the answer text only. Put your suggestions / \
opinions / analysis directly in the ``answer`` field; that becomes \
the spoken reply.

## Examples

(refs are illustrative; use actual refs from the current snapshot)

- User has selected paragraph e8, says "Review this." → \
``start_review(answer="Reviewing this paragraph.", paragraph_ref="e8", paragraph_text="The asynchronous-first model that emerged...")``
- "Add a note that this is too dense" with paragraph e8 selected → \
``reply(answer="Noted.", fills=[{"ref": "<textarea_ref>", "value": "This paragraph is too dense."}], click=["<save_button_ref>"])``
- "Where does it talk about rhythms?" → \
``reply(answer="Here, in this paragraph.", scroll_to="e14", select_text="e14")``"""


# ─────────────────────────────────────────────────────────────────────
# Peer workers: simulated reviewers that compute simple text metrics and
# send back a plausible-sounding review. The analysis is canned but
# varies per paragraph based on actual properties of the text.
# ─────────────────────────────────────────────────────────────────────


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


class ReviewWorker(ReplyToolMixin, UIWorker):
    """UIWorker that drives the document review workspace.

    Composes ``ReplyToolMixin`` for the bundled reply tool and adds a
    ``start_review`` tool for kicking off paragraph review. A
    ``@ui_event("note_click")`` handler converts client-side note
    clicks into ``select_text`` navigation. ``on_job_response`` is
    overridden to translate each reviewer's response into an ``add_note``
    UI command so feedback shows up in the notes panel as it lands.

    ``keep_history=True`` so the worker can resolve deixis like "can we
    add a note for that?" against its own prior replies.
    """

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=UI_PROMPT),
        )
        super().__init__("ui", llm=llm, keep_history=True)
        # job_id -> {"paragraph_ref": "..."}; lets on_job_response know
        # which paragraph a reviewer's feedback belongs to.
        self._reviews: dict[str, dict] = {}

    @tool
    async def start_review(
        self,
        params: FunctionCallParams,
        answer: str,
        paragraph_ref: str,
        paragraph_text: str,
    ):
        """Kick off a parallel review of one paragraph.

        Spawns the clarity and tone workers via ``start_ui_job_group``.
        Workers run in the background; their progress is forwarded to the
        page automatically. As each completes, ``on_job_response``
        translates the response into an ``add_note`` UI command.

        Args:
            answer: A short spoken acknowledgement ("Reviewing this
                paragraph").
            paragraph_ref: The snapshot ref of the paragraph under
                review.
            paragraph_text: The paragraph's text content. Workers analyze
                this directly.
        """
        logger.info(f"{self}: start_review(ref={paragraph_ref!r})")
        job_id = await self.start_ui_job_group(
            "clarity",
            "tone",
            payload={"ref": paragraph_ref, "text": paragraph_text},
            label=f"Reviewing ¶ {paragraph_ref}",
        )
        # Remember which paragraph this review is for so we can attach
        # each worker's response to the right note.
        self._reviews[job_id] = {"paragraph_ref": paragraph_ref}
        await self.respond_to_job(answer, tts_speak=True)
        await params.result_callback(None)

    async def on_job_response(self, message: BusJobResponseMessage) -> None:
        """Turn reviewer responses into ``add_note`` UI commands."""
        await super().on_job_response(message)
        review = self._reviews.get(message.job_id)
        if not review:
            return
        if message.status != JobStatus.COMPLETED:
            return
        feedback = ((message.response or {}).get("feedback") or "").strip()
        if not feedback:
            return
        await self.send_command(
            "add_note",
            {
                "source": message.source,
                "ref": review["paragraph_ref"],
                "text": feedback,
            },
        )

    @ui_event("note_click")
    async def on_note_click(self, message) -> None:
        """User clicked a note in the panel; jump to its paragraph."""
        ref = (message.payload or {}).get("ref")
        if not isinstance(ref, str) or not ref:
            return
        logger.info(f"{self}: note_click → scroll_to + select_text({ref!r})")
        await self.scroll_to(ref)
        await self.select_text(ref)


@tool_options(cancel_on_interruption=False, timeout_secs=30)
async def answer_about_screen(params: FunctionCallParams, query: str):
    """Forward the user's request to the screen-aware review worker.

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
    logger.info("Starting document-review bot")

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

    await runner.add_workers(
        ReviewWorker(),
        ClarityReviewer("clarity"),
        ToneReviewer("tone"),
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
