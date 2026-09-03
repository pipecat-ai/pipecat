#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Backend LLM worker for two-tier voice agents.

A frontend holds the conversation and hands off requests that need tools or
careful reasoning. What the frontend is does not matter to this contract: a
speech-to-speech model delegating on its own, or a pipeline calling a tool.
A :class:`BackendLLMWorker` runs any Pipecat LLM service, with its own context
and multi-step tool calling, to do that work: over the worker job API it
streams back everything it produces and returns its final answer.
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


@dataclass
class BackendOutput:
    """One piece of output from a backend, on its way to the frontend.

    Parameters:
        text: The text the backend produced.
        is_thought: Whether this is a reasoning summary rather than a response.
        is_final: Whether this is the backend's answer to the task, as opposed
            to progress on the way to it.
        speakable: Whether the user may hear this. It is not a promise that
            the frontend speaks it: a frontend decides what to do with the
            flag, and a speech-to-speech model may paraphrase or skip it.
    """

    text: str
    is_thought: bool = False
    is_final: bool = False
    speakable: bool = True

    def to_payload(self) -> dict[str, Any]:
        """Render the output as a job update payload.

        Returns:
            The payload.
        """
        return {
            "text": self.text,
            "is_thought": self.is_thought,
            "is_final": self.is_final,
            "speakable": self.speakable,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BackendOutput":
        """Rebuild an output from a job update payload.

        Args:
            payload: The update payload.

        Returns:
            The output.
        """
        return cls(
            text=str(payload.get("text") or ""),
            is_thought=bool(payload.get("is_thought")),
            is_final=bool(payload.get("is_final")),
            speakable=bool(payload.get("speakable")),
        )


#: Called with each piece of output a backend produces, as it is produced.
BackendUpdateCallback = Callable[[BackendOutput], Awaitable[None]]

#: Adjusts a backend output — its text, or whether the user may hear it —
#: before it leaves the worker.
BackendOutputTransform = Callable[[BackendOutput], Awaitable[BackendOutput]]


def render_backend_request(task: str | None, messages: list[dict[str, Any]], *, first: bool) -> str:
    """Render a delegated task, with the conversation that led to it, as one user message.

    The conversation is labelled transcript text rather than context messages
    of their own, so the backend's context holds only what the backend itself
    said as assistant messages.

    Args:
        task: The request the frontend delegated, when it worded one. Without
            it the backend is asked to act on the conversation.
        messages: Fragments of the voice conversation not yet seen by the
            backend, as ``{"role", "content"}`` dicts.
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
    if task:
        lines.append(f"Task from the voice assistant: {task}")
    else:
        lines.append("Act on the user's most recent request in the conversation above.")
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
    context as one user message (the conversation since the previous task,
    then the task), and runs the LLM until it produces a final answer.

    Job contract (``@job(name="run")``, one task at a time):

    - request payload: ``{"task": str | None, "messages": [{"role": "user" |
      "assistant", "content": str}, ...]}``. A frontend that words the request
      itself sends a ``task``; one whose model hands over without wording
      anything sends only the conversation, and the backend works out what is
      being asked.
    - updates: a :class:`BackendOutput` payload for every piece of output —
      reasoning summaries, what the backend says before calling tools, and its
      final answer.
    - response: ``{"text": str}`` — the final answer, or ``""`` if the task
      ended without one.

    The final answer arrives twice, as the last update and as the response, so
    a caller uses one or the other: a frontend relaying output as it arrives
    reads the updates, while one that needs a return value (a tool handler,
    say) reads the response and skips updates marked ``is_final``.

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
        transform_output: BackendOutputTransform | None = None,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
    ):
        """Initialize the backend worker.

        Args:
            llm: The backend LLM service.
            context: The backend's context, typically carrying its tools.
                A fresh empty context when omitted.
            name: Worker name; auto-generated when omitted.
            transform_output: Called with each :class:`BackendOutput` before it
                is sent, to adjust its text or whether the user may hear it.
                Without one, responses are speakable and reasoning summaries
                are not.
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
        self._transform_output = transform_output

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
        task = str(payload.get("task") or "").strip() or None
        messages = payload.get("messages") or []
        if not task and not messages:
            logger.warning(
                f"Worker '{self.name}': job {message.job_id} has neither a task nor conversation"
            )
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
        if text:
            # Default behavior: only the final answer is speakable.
            # This behavior can be adjusted by a transform_output callback.
            await self._emit(run, BackendOutput(text=text, is_final=finished, speakable=finished))
        if finished:
            run.final_text = text
            run.finished.set()

    async def _on_assistant_thought(self, message: AssistantThoughtMessage):
        run = self._run
        text = (message.content or "").strip()
        if run is not None and text:
            await self._emit(run, BackendOutput(text=text, is_thought=True, speakable=False))

    async def _emit(self, run: "_BackendRun", output: BackendOutput):
        """Send one output as a job update, after any configured transform."""
        if self._transform_output is not None:
            output = await self._transform_output(output)
        if output.text:
            await self.send_job_update(run.job_id, output.to_payload())


async def run_backend_job(
    worker: BaseWorker,
    backend_name: str,
    *,
    task: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    on_update: BackendUpdateCallback | None = None,
    timeout_secs: float | None = None,
) -> str:
    """Send a task to a :class:`BackendLLMWorker` and return its final text.

    Args:
        worker: The worker making the request (for a pipeline processor,
            ``self.pipeline_worker``).
        backend_name: Name of the backend worker.
        task: The request to delegate, when the frontend words one. Omit it
            when the frontend hands over without saying what it wants, and the
            backend will work that out from ``messages``.
        messages: Fragments of the voice conversation the backend hasn't seen
            yet.
        on_update: Called with each :class:`BackendOutput` the backend
            produces, the final answer included. A caller using the return
            value should skip outputs marked ``is_final`` to avoid handling
            the answer twice.
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
            output = BackendOutput.from_payload(event.data)
            if output.text:
                await on_update(output)
    return str(backend_job.response.get("text", ""))
