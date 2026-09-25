#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shopping list: the voice LLM asks, the UI worker finds and acts.

The page renders a shopping list. The user talks: "add milk and eggs",
"check off the bread", "drop the last one", "what's left?". The voice LLM
understands the words and calls a tool with what it understood; the tool
sends a job to the UI worker, which finds the item on screen with its
classifier and sends the command. Nothing on the UI side runs an LLM turn.

- The **voice layer** is an ordinary voice pipeline (STT, LLM, TTS). Its
  LLM converses and calls two tools: ``update_list`` with everything the
  turn asked for, and ``check_list`` to read the list. It never sees the
  screen; both return short data.
- The **UI worker** ("ui") owns the list. Each tool is a job on it. For an
  item named in words, the worker asks its classifier which checkbox on
  the screen the words mean, then sends ``set_checked`` or
  ``remove_item``. ``add_item`` needs no classifier: the voice LLM already
  carries the text. ``check_list`` reads the snapshot with plain code.

With ``TYPESAFE_API_KEY`` set the worker uses Jev, which answers in about
a tenth of a second with a calibrated probability; otherwise the worker's
own LLM answers through an ``LLMClassifier``.

Architecture::

    Voice pipeline (PipelineWorker "main", owns transport + RTVI):
      transport.in -> STT -> user_agg -> LLM -> TTS -> transport.out -> assistant_agg
        └── @tool update_list(add, check, uncheck, remove, highlight, clear_checked)
            / check_list
              └── params.pipeline_worker.job("ui", name=..., payload=...)

    ListWorker (UIWorker "ui", no LLM turn):
      ├── @job update   -> add_item per text; the classifier picks the checkbox
      │                    for each item to check, uncheck, remove or highlight
      └── @job summary  -> the list from the snapshot

Run::

    uv run bot.py

Then open the client at ``http://localhost:5173`` (see ``README.md``).

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- TYPESAFE_API_KEY (optional, for Jev)
"""

import os
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus.messages import BusJobRequestMessage
from pipecat.classifiers.base_classifier import ChoiceQuestion
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
You are the voice of a shopping-list assistant. You cannot see the \
screen; the list lives there and your tools change and read it.

When the user wants the list changed, call update_list once with \
everything they asked for: items to add, check off, uncheck or remove, \
and clear_checked for "clear the ones I've got". Name each item as the \
thing itself, never as a pronoun: if you suggested jamón ibérico and \
the user says "add that", pass add=["jamón ibérico"]; "check off the \
last one" after you listed milk, eggs and bread is check=["bread"]. \
When you are not sure what "that" or "it" refers to, ask before \
calling the tool. For ANY question about the list, \
call check_list and answer only from what it returns; the user can edit \
the list on screen at any time, so call it again every time. When your \
answer names items, "the drinks are milk and juice", "you still need \
eggs", flash them with update_list(highlight=[...]) as you answer. If a tool \
says an item was not found, say so briefly and ask which one they mean.

Keep every reply to one short spoken sentence. Don't describe how \
you're updating the list; the screen shows that. For a plain greeting, \
greet back warmly."""


def _checkboxes(node: Any, items: list[tuple[str, str, bool]]) -> None:
    """Walk a snapshot node, collecting (ref, text, checked) for each checkbox."""
    if not isinstance(node, dict):
        return
    if node.get("role") == "checkbox":
        ref, name = node.get("ref"), node.get("name")
        if isinstance(ref, str) and isinstance(name, str) and name:
            items.append((ref, name, "checked" in (node.get("state") or [])))
    for child in node.get("children") or []:
        _checkboxes(child, items)


class ListWorker(UIWorker):
    """UIWorker that keeps the shopping list, with a classifier and no LLM turn.

    Every job takes the items in the user's words. The classifier picks the
    checkbox each one means among those on screen; adding needs no
    classifier, since the voice LLM already carries the text.
    """

    def __init__(self):
        llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"])
        api_key = os.getenv("TYPESAFE_API_KEY")
        classifier = JevClassifier(api_key=api_key) if api_key else None
        super().__init__(UI_NAME, llm=llm, classifier=classifier)

    @job(name="update")
    async def _update(self, message: BusJobRequestMessage) -> None:
        payload = message.payload or {}
        done: dict[str, list[str]] = {
            "added": [],
            "checked": [],
            "unchecked": [],
            "removed": [],
            "highlighted": [],
        }
        not_found: list[str] = []

        for text in _texts(payload.get("add")):
            await self.send_command("add_item", {"text": text})
            done["added"].append(text)

        for field, verb, command, extra in (
            ("check", "checked", "set_checked", {"checked": True}),
            ("uncheck", "unchecked", "set_checked", {"checked": False}),
            ("remove", "removed", "remove_item", {}),
            ("highlight", "highlighted", "highlight", {}),
        ):
            for text in _texts(payload.get(field)):
                found = await self._item(text)
                if not found:
                    not_found.append(text)
                    continue
                ref, label = found
                await self.send_command(command, {"ref": ref, **extra})
                done[verb].append(label)

        if payload.get("clear_checked"):
            for ref, name, checked in self._list():
                if checked:
                    await self.send_command("remove_item", {"ref": ref})
                    done["removed"].append(name)

        await self.send_job_response(message.job_id, {**done, "not_found": not_found})

    @job(name="summary")
    async def _summary(self, message: BusJobRequestMessage) -> None:
        items = [{"item": name, "checked": checked} for _, name, checked in self._list()]
        await self.send_job_response(message.job_id, {"items": items})

    def _list(self) -> list[tuple[str, str, bool]]:
        """The list on screen as (ref, text, checked), in page order."""
        items: list[tuple[str, str, bool]] = []
        _checkboxes((self.snapshot or {}).get("root"), items)
        return items

    async def _item(self, text: str) -> tuple[str, str] | None:
        """The checkbox the user's words mean, as (ref, label), or None."""
        items = self._list()
        if not items:
            return None
        options: dict[str, Any] = {ref: name for ref, name, _ in items}
        question = ChoiceQuestion(instructions="the list item the user means", options=options)
        result = (await self.classifier.choice(text, {"item": question}))["item"]
        logger.debug(
            f"{self.name}: {text!r} -> {options[result.choice]!r} ({result.confidence:.2f})"
        )
        if result.confidence < 0.5:
            return None
        return result.choice, options[result.choice]


def _texts(items: Any) -> list[str]:
    """The non-empty strings in a payload field, or none when it is missing."""
    if not isinstance(items, list):
        return []
    return [i.strip() for i in items if isinstance(i, str) and i.strip()]


async def _ui(params: FunctionCallParams, name: str, payload: dict) -> None:
    """Send a job to the UI worker and hand its answer to the voice LLM."""
    try:
        async with params.pipeline_worker.job(
            UI_NAME, params=JobParams(name=name, payload=payload, timeout=15)
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"ui job {name} failed: {e}")
        await params.result_callback({"error": str(e)})
        return
    await params.result_callback(t.response)


@tool_options(cancel_on_interruption=False, timeout_secs=15)
async def update_list(
    params: FunctionCallParams,
    add: list[str] | None = None,
    check: list[str] | None = None,
    uncheck: list[str] | None = None,
    remove: list[str] | None = None,
    highlight: list[str] | None = None,
    clear_checked: bool = False,
):
    """Change the shopping list, or point at items on it.

    One call covers everything the user asked for in a turn. Name each item
    as the thing itself, never as a pronoun: "add that" after you suggested
    jamón ibérico is add=["jamón ibérico"], and "check off the last one"
    after you listed milk, eggs and bread is check=["bread"]. Highlight the
    items you are talking about when you answer a question about the list,
    such as the drinks or what is left. The answer says what was done and
    which items were not found on the list.

    Args:
        params: Framework-provided tool invocation context.
        add: Items to add, named as they should appear on the list.
        check: Items to check off, as on the list or as the user said them.
        uncheck: Items to put back on the list.
        remove: Items to remove.
        highlight: Items to flash on screen, to show what you are talking about.
        clear_checked: Remove every item that is checked off.
    """
    payload: dict = {}
    for field, value in (
        ("add", add),
        ("check", check),
        ("uncheck", uncheck),
        ("remove", remove),
        ("highlight", highlight),
    ):
        if value:
            payload[field] = value
    if clear_checked:
        payload["clear_checked"] = True
    await _ui(params, "update", payload)


@tool_options(cancel_on_interruption=False, timeout_secs=15)
async def check_list(params: FunctionCallParams):
    """Look up what is on the shopping list and what is checked off.

    Call it for any question about the list. The user can edit the list on
    screen at any time, so earlier results are stale; call it again every
    time rather than answering from memory.

    Args:
        params: Framework-provided tool invocation context.
    """
    await _ui(params, "summary", {})


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting shopping-list bot")

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

    context = LLMContext(tools=[update_list, check_list])
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

    await runner.add_workers(ListWorker(), worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user briefly. Tell them they can build their "
                    "shopping list by voice: add items, check things off, or "
                    "ask what's left. One short sentence."
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
