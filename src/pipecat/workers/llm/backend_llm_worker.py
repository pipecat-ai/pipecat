#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Backend LLM worker for two-tier voice agents.

A frontend conversational model — a speech-to-speech model such as OpenAI Live,
or a fast cascade LLM — hands off requests that need tools or careful
reasoning. A :class:`BackendLLMWorker` runs any Pipecat LLM service, with its
own context and multi-step tool calling, to do that work: over the worker job
API it streams back what it says along the way and returns its final answer.
:func:`run_backend_job` is the caller side of that contract.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.bus.messages import BusJobRequestMessage
from pipecat.frames.frames import LLMContextFrame, LLMMessagesAppendFrame
from pipecat.pipeline.job_context import JobEvent, JobParams
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantThoughtMessage,
    AssistantTurnStoppedMessage,
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm.llm_context_worker import LLMContextWorker

#: Name of the job a :class:`BackendLLMWorker` handles.
BACKEND_JOB_NAME = "run"

#: Called with ``(kind, text)`` for each update a backend streams back while it works.
BackendUpdateCallback = Callable[[str, str], Awaitable[None]]


def render_backend_request(task: str, messages: list[dict[str, Any]], *, first: bool) -> str:
    """Render a delegated task, with the voice turns that led to it, as one user message.

    The turns are labelled transcript text rather than context messages of
    their own, so the backend's context holds only what the backend itself
    said as assistant messages.

    Args:
        task: The request the frontend delegated.
        messages: User/assistant turns of the voice conversation not yet seen
            by the backend, as ``{"role", "content"}`` dicts.
        first: Whether this is the backend's first task in the conversation.

    Returns:
        The message text.
    """
    lines: list[str] = []
    if messages:
        lines.append(
            "Voice conversation so far:" if first else "Voice conversation since your last task:"
        )
        for message in messages:
            content = message.get("content")
            if content:
                lines.append(f"{str(message.get('role', '')).upper()}: {content}")
        lines.append("")
    lines.append(f"Task from the voice assistant: {task}")
    return "\n".join(lines)


@dataclass
class _BackendRun:
    """A job in progress: one delegated task and the LLM runs it takes."""

    job_id: str
    runs_requested: int = 0
    runs_completed: int = 0
    final_text: str = ""
    finished: asyncio.Event = field(default_factory=asyncio.Event)


class BackendLLMWorker(LLMContextWorker):
    """A worker that runs an LLM service as the backend a frontend delegates to.

    The worker owns the backend's conversation: an ``LLMContext`` plus the
    aggregator pair, so multi-step tool calling works as it does in any
    pipeline. Each delegated task arrives as a ``run`` job, is appended to the
    context as one user message (the voice turns since the previous task, then
    the task), and runs the LLM until it produces a final answer. Responses
    along the way — what the backend says before calling tools — and its
    reasoning summaries are streamed back as job updates; the final answer is
    the job response.

    Job contract (``@job(name="run")``, one task at a time):

    - request payload: ``{"task": str, "messages": [{"role": "user" | "assistant",
      "content": str}, ...]}``
    - updates: ``{"kind": "text" | "thought", "text": str}`` — ``text`` is an
      intermediate assistant response, suitable for the frontend to speak
      while the backend keeps working; ``thought`` is a reasoning summary
      (from the LLM's thought frames), progress the frontend may draw on but
      not speak
    - response: ``{"text": str}`` — the final assistant response, or ``""`` if
      the task ended without one

    :func:`run_backend_job` wraps the caller side.

    Example::

        backend = BackendLLMWorker(
            llm=AnthropicLLMService(api_key=...),
            context=LLMContext(
                [{"role": "system", "content": BACKEND_INSTRUCTIONS}],
                tools=[get_weather],
            ),
        )
    """

    def __init__(
        self,
        *,
        llm: LLMService[Any],
        context: LLMContext | None = None,
        name: str | None = None,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
    ):
        """Initialize the backend worker.

        Args:
            llm: The backend LLM service.
            context: The backend's context, typically carrying its system
                instructions and tools. A fresh empty context when omitted.
            name: Worker name; auto-generated when omitted.
            user_params: Optional parameters for the user aggregator. Defaults
                to external turn strategies: the backend has no audio, so the
                default VAD and turn-analysis strategies (and the model the
                latter loads) have nothing to do here.
            assistant_params: Optional parameters for the assistant aggregator.
        """
        if user_params is None:
            user_params = LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies())
        super().__init__(
            name,  # type: ignore[arg-type]  # None selects an auto-generated name
            llm=llm,
            active=True,
            context=context,
            user_params=user_params,
            assistant_params=assistant_params,
        )
        self._run: _BackendRun | None = None
        self._jobs_run = 0

        # A task takes one or more LLM runs: the first for the task itself,
        # then one per round of tool results (the assistant aggregator pushes
        # an LLMContextFrame back to the LLM after each). Runs are counted as
        # the LLM picks them up and responses as they end; the task is
        # finished when the two match and no further run is on the way.
        @self.llm.event_handler("on_before_process_frame")
        async def on_before_process_frame(llm, frame):
            if isinstance(frame, LLMContextFrame) and self._run is not None:
                self._run.runs_requested += 1

        @self.assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            await self._on_assistant_turn_stopped(message)

        @self.assistant_aggregator.event_handler("on_assistant_thought")
        async def on_assistant_thought(aggregator, message: AssistantThoughtMessage):
            await self._on_assistant_thought(message)

    @job(name=BACKEND_JOB_NAME, sequential=True)
    async def run_task(self, message: BusJobRequestMessage):
        """Run one delegated task to completion, streaming intermediate responses as updates.

        Args:
            message: The job request; see the class docstring for the payload.
        """
        payload = message.payload or {}
        task = str(payload.get("task") or "").strip()
        messages = payload.get("messages") or []
        if not task:
            logger.warning(f"Worker '{self.name}': job {message.job_id} has no task")
            await self.send_job_response(message.job_id, {"text": ""})
            return

        self._jobs_run += 1
        run = self._run = _BackendRun(job_id=message.job_id)
        text = render_backend_request(task, messages, first=self._jobs_run == 1)
        await self.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True)
        )
        try:
            await run.finished.wait()
        finally:
            self._run = None
        await self.send_job_response(message.job_id, {"text": run.final_text})

    async def _on_assistant_turn_stopped(self, message: AssistantTurnStoppedMessage):
        run = self._run
        if run is None:
            return
        run.runs_completed += 1
        # A further run is on the way while a tool call is in flight, or while
        # a re-run is queued that the LLM hasn't picked up yet (a tool that
        # returns before its response ends queues one early).
        finished = (
            run.runs_completed >= run.runs_requested
            and not self.assistant_aggregator.has_function_calls_in_progress
            and not self.llm.has_queued_frame(LLMContextFrame)
        )
        text = (message.content or "").strip()
        if finished:
            run.final_text = text
            run.finished.set()
        elif text:
            await self.send_job_update(run.job_id, {"kind": "text", "text": text})

    async def _on_assistant_thought(self, message: AssistantThoughtMessage):
        run = self._run
        text = (message.content or "").strip()
        if run is not None and text:
            await self.send_job_update(run.job_id, {"kind": "thought", "text": text})


async def run_backend_job(
    worker: BaseWorker,
    backend_name: str,
    *,
    task: str,
    messages: list[dict[str, Any]] | None = None,
    on_update: BackendUpdateCallback | None = None,
    timeout_secs: float | None = None,
) -> str:
    """Send a task to a :class:`BackendLLMWorker` and return its final text.

    Args:
        worker: The worker making the request (for a pipeline processor,
            ``self.pipeline_worker``).
        backend_name: Name of the backend worker.
        task: The request to delegate.
        messages: Voice conversation turns the backend hasn't seen yet.
        on_update: Called with ``(kind, text)`` for each intermediate response
            (``"text"``) or reasoning summary (``"thought"``) the backend
            streams back while it works; the final response is returned, not
            passed here.
        timeout_secs: How long to wait for the backend, including the wait for
            it to become ready.

    Returns:
        The backend's final response text (``""`` if it produced none).

    Raises:
        JobError: If the backend fails, is cancelled, or times out.
    """
    params = JobParams(
        name=BACKEND_JOB_NAME,
        payload={"task": task, "messages": messages or []},
        timeout=timeout_secs,
    )
    async with worker.job(backend_name, params=params) as backend_job:
        async for event in backend_job:
            if event.type != JobEvent.UPDATE or not event.data or not on_update:
                continue
            text = event.data.get("text")
            if text:
                await on_update(str(event.data.get("kind", "text")), str(text))
    return str(backend_job.response.get("text", ""))
