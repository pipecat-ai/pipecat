#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shopping list — every voice turn drives the UI; speech is incidental.

The page renders a shopping list. The user talks: "add milk and eggs",
"check off the bread", "drop the last one", "what's left?". Every user
turn updates the list on screen; the assistant may also say something
back. Updates run in parallel on separate workers — the voice layer
never mutates the list.

This is the pattern for "every input acts, may speak":

- The **voice layer** is an ordinary voice pipeline (STT → LLM → TTS).
  Its LLM converses and never *mutates* the list. It has one read-only
  tool, ``check_list``, to look up what's currently on screen when the
  user asks — otherwise it can't see the page, and so can't know about
  items the user checked off (or added) by hand.
- A **UIWorker** ("ui") does all the list work. It is *not* in the voice
  pipeline. Instead, the voice pipeline's user aggregator fires
  ``on_user_turn_stopped`` once per user turn; that handler dispatches
  the transcript to the UIWorker as a ``respond`` job (a bus message).
  The UIWorker reads the current ``<ui_state>`` snapshot (auto-injected
  before its inference) and calls ``update_list`` to add / check /
  remove items. It acts silently — its LLM output never reaches TTS,
  because it lives on its own worker.

The snapshot is the shared source of truth: the UIWorker acts on it, and
``check_list`` reads it (via ``ListWorker.list_summary``) so the voice
layer answers list questions from what's really on screen — including
manual edits — instead of from conversation memory.

Architecture::

    Voice pipeline (PipelineWorker "main", owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        ├── @on_user_turn_stopped: worker.job("ui", name="respond",
        │                                     payload={"query": transcript})
        └── @tool check_list() → list_worker.list_summary()   # read-only

    UIWorker "ui" (job-based, acts silently):
      └── @tool update_list(add, check, uncheck, remove, highlight)
            └── send_command("add_item" / "set_checked" / "remove_item")
                + respond_to_job()   # no answer (silent)
      └── list_summary() → reads the live snapshot for check_list

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
You are the voice of a shopping-list assistant. A separate UI layer \
sees the list and updates it on screen; you do NOT touch the list \
yourself and you cannot see the screen.

Be a brief, friendly companion. Acknowledge what the user asked for \
("Sure, adding milk and eggs") and react naturally. When the user asks \
what's on the list or what's still needed, call ``check_list`` and \
answer from what it returns — the user may have checked items off (or \
added them) on screen themselves, so never answer about the list's \
contents from memory. Keep every reply to one short spoken sentence. \
Don't describe how you're updating the list — the screen shows that. \
For a plain greeting, greet back warmly."""


# The UI wire-format guide (UI_STATE_PROMPT_GUIDE) is appended to the LLM's
# system instruction automatically by UIWorker, so this prompt only needs the
# app-specific behavior.
UI_PROMPT = """\
You maintain a shopping list by voice. The current ``<ui_state>`` \
block is in your context. Each list item is a checkbox with a \
snapshot ref (e.g. ``e5``) and a label (the item text). A checked-off \
item is tagged ``[checked]`` (unchecked items have no such tag). Use \
the labels to resolve which item the user means, and the ``[checked]`` \
tag to tell done from not-done.

## Tool: update_list

Every turn calls ``update_list`` exactly once — even when there's \
nothing to change (call it with no arguments). One tool call per turn.

``update_list(add=None, check=None, uncheck=None, remove=None, highlight=None)``:

- ``add`` (OPTIONAL): a list of item texts to append. Split a spoken \
list into separate items ("milk, eggs and bread" → \
``["milk", "eggs", "bread"]``). Normalize lightly (lowercase, drop \
filler like "some" / "a couple of").
- ``check`` (OPTIONAL): a list of refs to mark done.
- ``uncheck`` (OPTIONAL): a list of refs to mark not done.
- ``remove`` (OPTIONAL): a list of refs to delete.
- ``highlight`` (OPTIONAL): a list of refs to flash briefly. Use it to \
*show* the user something when they ask (e.g. highlight the unchecked \
items for "what's left?").

## Decision rules

- **"Add X (and Y)"** → ``add`` with one entry per item.
- **"Check off / got / cross out X"** → resolve X's ref from \
``<ui_state>``, set ``check``.
- **"Uncheck / put back X"** → set ``uncheck``.
- **"Remove / delete / never mind X"**, **"clear the checked ones"** → \
set ``remove`` with the matching refs (read the checked state from \
``<ui_state>`` for "the checked ones").
- **"the last one"** → the last item in ``<ui_state>``.
- **"What's left? / what do I still need?"** → ``highlight`` the \
unchecked items. **"What's on my list?"** → ``highlight`` everything. \
(The voice layer speaks; you just point.)
- **A greeting or anything not about the list** → call ``update_list`` \
with no arguments (no-op) so the turn completes.

## Examples

(refs are illustrative; use the actual refs from the current \
``<ui_state>``)

- "Add milk and a dozen eggs." → \
``update_list(add=["milk", "eggs"])``
- "Check off the milk." → ``update_list(check=["e5"])``
- "Actually drop the eggs." → ``update_list(remove=["e7"])``
- "Clear the ones I've got." (e5, e9 checked) → \
``update_list(remove=["e5", "e9"])``
- "What's left?" (e7, e11 unchecked) → \
``update_list(highlight=["e7", "e11"])``
- "Hey there." → ``update_list()``"""


def _collect_checkbox_items(node, items: list[tuple[str, bool]]) -> None:
    """Walk an a11y snapshot node, collecting (text, checked) for each checkbox."""
    if not isinstance(node, dict):
        return
    if node.get("role") == "checkbox":
        name = node.get("name")
        if isinstance(name, str) and name:
            items.append((name, "checked" in (node.get("state") or [])))
    for child in node.get("children") or []:
        _collect_checkbox_items(child, items)


class ListWorker(UIWorker):
    """UIWorker that maintains the shopping list, silently.

    A single bundled ``update_list`` tool, always called once per turn,
    maps the user's request to ``add_item`` / ``set_checked`` /
    ``remove_item`` UI commands (plus the standard ``highlight``). The
    tool completes the ``respond`` job with no answer
    (``respond_to_job()``), so nothing is spoken — the separate voice layer
    owns speech.
    """

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=UI_PROMPT),
        )
        super().__init__(UI_NAME, llm=llm)

    @tool
    async def update_list(
        self,
        params: FunctionCallParams,
        add: list[str] | None = None,
        check: list[str] | None = None,
        uncheck: list[str] | None = None,
        remove: list[str] | None = None,
        highlight: list[str] | None = None,
    ):
        """Update the shopping list. Called exactly once per turn.

        Args:
            add: New item texts to append (one entry per item).
            check: Snapshot refs of items to mark done.
            uncheck: Snapshot refs of items to mark not done.
            remove: Snapshot refs of items to delete.
            highlight: Snapshot refs of items to flash briefly (e.g. to
                show what's left when the user asks).
        """
        logger.info(
            f"{self}: update_list(add={add!r}, check={check!r}, uncheck={uncheck!r}, "
            f"remove={remove!r}, highlight={highlight!r})"
        )
        # Defensive guards: skip malformed entries so a stray value can't
        # crash the tool before respond_to_job fires (which would hold the
        # single-flight lock until the requester's timeout).
        for text in add or []:
            if isinstance(text, str) and text.strip():
                await self.send_command("add_item", {"text": text.strip()})
        for ref in check or []:
            if isinstance(ref, str):
                await self.send_command("set_checked", {"ref": ref, "checked": True})
        for ref in uncheck or []:
            if isinstance(ref, str):
                await self.send_command("set_checked", {"ref": ref, "checked": False})
        for ref in remove or []:
            if isinstance(ref, str):
                await self.send_command("remove_item", {"ref": ref})
        for ref in highlight or []:
            if isinstance(ref, str):
                await self.highlight(ref)
        await self.respond_to_job()
        await params.result_callback(None)

    def list_summary(self) -> str:
        """Summarize the current list from the live ``<ui_state>`` snapshot.

        Reads the same snapshot the worker acts on, so it reflects
        everything on screen — including items the user checked off (or
        added) by hand. The voice layer's ``check_list`` tool calls this
        to answer list questions from ground truth, not conversation memory.
        """
        items: list[tuple[str, bool]] = []
        _collect_checkbox_items((self._latest_snapshot or {}).get("root"), items)
        if not items:
            return "The shopping list is empty."
        entries = [
            f"{text} ({'checked off' if checked else 'still needed'})" for text, checked in items
        ]
        return "Current list: " + "; ".join(entries) + "."


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting shopping-list bot")

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

    # The UI worker owns the list; create it up front so the voice layer's
    # read-only ``check_list`` tool can look at its live snapshot. It's added
    # to the runner at the end alongside the main worker.
    list_worker = ListWorker()

    @tool_options(timeout_secs=10)
    async def check_list(params: FunctionCallParams):
        """Look up what's currently on the shopping list and what's checked off.

        Call this whenever the user asks what's on their list or what's
        still needed. The list reflects items the user may have checked
        off (or added) on screen themselves, so don't answer about its
        contents from memory.
        """
        summary = list_worker.list_summary()
        logger.info(f"check_list -> {summary!r}")
        await params.result_callback(summary)

    context = LLMContext(tools=[check_list])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
    user_aggregator = aggregators.user()
    assistant_aggregator = aggregators.assistant()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Every user turn drives the UI: forward the transcript to the UIWorker
    # as a respond job (a bus message). Fire it from the turn-stopped event
    # so the voice LLM (which runs from the same turn) and the UIWorker act
    # in parallel — list *changes* never go through a tool call (the voice
    # LLM only uses a tool to *read* the list, via check_list). This handler
    # runs in its own task, so awaiting the job here does not block the voice path.
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        transcript = (message.content or "").strip()
        if not transcript:
            return
        logger.info(f"Dispatching turn to UI worker: {transcript!r}")
        try:
            async with worker.job(
                UI_NAME, name="respond", payload={"query": transcript}, timeout=15
            ):
                pass
        except JobError as e:
            logger.warning(f"ui job failed: {e}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user briefly. Tell them they can build their "
                    "shopping list by voice — add items, check things off, or "
                    "ask what's left. One short sentence."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(list_worker, worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
