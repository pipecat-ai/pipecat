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
streams back everything it produces, its final answer last.
:func:`delegate_to_backend` is the caller side of that contract, an async
iterator over that stream.

A request is text, so how a frontend words one is its own business.
:func:`render_transcript_request` renders the conversation as a labelled
transcript, which is what a frontend hands over when its model signals a
handoff without wording a request; a frontend whose model does word one sends
that instead.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.bus.messages import BusJobRequestMessage
from pipecat.frames.frames import ErrorFrame, LLMContextFrame, LLMMessagesAppendFrame
from pipecat.pipeline.job_context import JobEvent, JobParams, JobStatus
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    LLMStandardMessage,
)
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantThoughtMessage,
    AssistantTurnStoppedMessage,
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm.llm_context_worker import LLMContextWorker

#: Name of the job a :class:`BackendLLMWorker` handles.
BACKEND_JOB_NAME = "run"

#: The ``type`` of a job update that carries a :class:`BackendOutput`. Other
#: update types may share the stream; :func:`delegate_to_backend` yields only
#: outputs.
OUTPUT_UPDATE_TYPE = "output"


@dataclass
class BackendOutput:
    """One piece of output from a backend, on its way to the frontend.

    Parameters:
        text: The text the backend produced.
        is_thought: Whether this is a reasoning summary rather than a response.
        is_final: Whether this is the backend's answer to the delegation, as opposed
            to progress on the way to it.
        prefers_spoken: Whether the backend would like the user to hear this.
            This is just a hint to the frontend: it may choose to follow it or not.
            ``OpenAILiveLLMService``'s live model takes it into consideration,
            but ultimately decides what to speak (or not) based on the conversation.
    """

    text: str
    is_thought: bool = False
    is_final: bool = False
    prefers_spoken: bool = True

    def to_payload(self) -> dict[str, Any]:
        """Render the output as a job update payload.

        Returns:
            The payload.
        """
        return {
            "type": OUTPUT_UPDATE_TYPE,
            "text": self.text,
            "is_thought": self.is_thought,
            "is_final": self.is_final,
            "prefers_spoken": self.prefers_spoken,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BackendOutput":
        """Rebuild an output from a job update payload.

        Args:
            payload: The update payload.

        Returns:
            The output.
        """
        # A flag the payload omits is left to the field's default, so the
        # defaults live in one place; one it carries is coerced, since a
        # payload can arrive from another process.
        flags = {
            name: bool(payload[name])
            for name in ("is_thought", "is_final", "prefers_spoken")
            if name in payload
        }
        return cls(text=str(payload.get("text") or ""), **flags)


#: Adjusts a backend output — its text, or whether the user may hear it —
#: before it leaves the worker.
BackendOutputTransform = Callable[[BackendOutput], Awaitable[BackendOutput]]


def message_text(message: LLMStandardMessage) -> str:
    """Return a context message's text, joining the text parts of list content.

    Text is all the transcript carries today, so non-text parts are left out.

    Args:
        message: A standard context message.

    Returns:
        The text, or ``""`` for a message that carries none.
    """
    content = message.get("content")  # type: ignore[attr-defined]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        )
    return ""


#: What :func:`render_transcript_request` tells the backend to do with a transcript.
DEFAULT_TRANSCRIPT_INSTRUCTION = "Act on the user's most recent request in the conversation above."


def render_transcript_request(
    conversation: Sequence[LLMContextMessage],
    *,
    instruction: str = DEFAULT_TRANSCRIPT_INSTRUCTION,
    first: bool = True,
) -> str:
    """Render a conversation as a labelled transcript for the backend to act on.

    Flattening the conversation into one message keeps the two conversations
    apart: the backend's context holds only what the backend itself said as
    assistant messages, so it never mistakes the frontend's speech for its
    own.

    Only user and assistant text crosses today: tool calls, tool results,
    images, other non-text content and messages in a service's own format are
    skipped, and say so at debug level. Carrying more is a question of how to
    render it.

    Args:
        conversation: The conversation the user is having with the frontend,
            as standard context messages.
        instruction: What the backend should do with the transcript, placed
            after it.
        first: Whether this is the backend's first request in the
            conversation. It decides how the transcript is introduced: as the
            conversation so far, or as what has been said since the previous
            delegation.

    Returns:
        The rendered request.
    """
    lines: list[str] = []
    if conversation:
        lines.append(
            "Voice conversation so far:"
            if first
            else "Voice conversation since the previous delegation:"
        )
        for message in conversation:
            if isinstance(message, LLMSpecificMessage):
                logger.debug(f"Skipping delegated message in {message.llm} format")
                continue
            role = message.get("role")
            text = message_text(message)
            if role in ("user", "assistant") and text:
                lines.append(f"{str(role).upper()}: {text}")
            else:
                logger.debug(f"Skipping delegated message with no transcript form: role={role!r}")
        lines.append("")
    lines.append(instruction)
    return "\n".join(lines)


@dataclass
class _BackendRun:
    """A job in progress: one delegation and the LLM runs it takes."""

    job_id: str
    runs_requested: int = 0
    runs_completed: int = 0
    final_text: str = ""
    error: str = ""
    finished: asyncio.Event = field(default_factory=asyncio.Event)


class BackendLLMWorker(LLMContextWorker):
    """A worker that runs an LLM service as the backend a frontend delegates to.

    The worker owns the backend's conversation: an ``LLMContext`` plus the
    aggregator pair, so multi-step tool calling works as it does in any
    pipeline. Each delegation arrives as a ``run`` job carrying the request
    text, which is appended to the context as one user message, and runs the
    LLM until it produces a final answer.

    Job contract (``@job(name="run")``, one delegation at a time):

    - request payload: ``{"request": str}`` — the text to put to the backend,
      composed by the frontend. What that text says is the application's
      business: :func:`render_transcript_request` renders the conversation as
      a transcript, which is what a frontend whose model hands off without
      wording a request needs, but a frontend that has a worded request can
      simply send it.
    - updates: a :class:`BackendOutput` payload for every piece of output —
      reasoning summaries, what the backend says before calling tools, and its
      final answer.
    - response: ``{"text": str}`` — the final answer, or ``""`` if the
      delegation ended without one. A backend LLM failure answers the job
      with ``JobStatus.ERROR``, so the frontend hears about it as soon as it
      happens.

    The final answer arrives twice, as the last update and as the response, so
    a caller uses one or the other. :func:`delegate_to_backend` wraps the
    caller side and reads the updates.

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
                Without one, only the final answer asks to be spoken: a
                frontend filling the wait is usually mid-sentence when
                progress arrives, and speaking it talks over them.
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
        self._transform_output = transform_output

        # A delegation takes one or more LLM runs: the first for the request
        # itself, then one per round of tool results (the assistant aggregator
        # pushes an LLMContextFrame back to the LLM after each). Runs are
        # counted as the LLM picks them up and responses as they end; the
        # delegation is finished when the two match and no further run is on
        # the way.
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

        @self.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame: ErrorFrame):
            await self._on_pipeline_error(frame)

    @job(name=BACKEND_JOB_NAME, sequential=True)
    async def run_delegation(self, message: BusJobRequestMessage):
        """Run one delegation to completion, streaming what the backend produces as updates.

        Args:
            message: The job request; see the class docstring for the payload.
        """
        request = str((message.payload or {}).get("request") or "").strip()
        if not request:
            logger.warning(f"Worker '{self.name}': job {message.job_id} has no request")
            await self.send_job_response(message.job_id, {"text": ""})
            return

        run = self._run = _BackendRun(job_id=message.job_id)
        await self.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": request}], run_llm=True)
        )
        try:
            await run.finished.wait()
        finally:
            self._run = None
        # An error fails the job only when it left the backend with nothing to say.
        if run.error and not run.final_text:
            logger.warning(f"Worker '{self.name}': job {message.job_id} failed: {run.error}")
            await self.send_job_response(message.job_id, {"text": ""}, status=JobStatus.ERROR)
            return
        await self.send_job_response(message.job_id, {"text": run.final_text})

    async def _on_assistant_turn_stopped(self, message: AssistantTurnStoppedMessage):
        run = self._run
        if run is None:
            return
        run.runs_completed += 1
        # A further run is on the way while a tool call is in flight, or while
        # a re-run is queued that the LLM hasn't picked up yet (a tool that
        # returns before its response ends queues one early). The check assumes
        # a settled tool call leaves the LLM something more to do: a tool that
        # returns no result, or passes run_llm=False, runs nothing further, so
        # the delegation waits out the caller's timeout.
        finished = (
            run.runs_completed >= run.runs_requested
            and not self.assistant_aggregator.has_function_calls_in_progress
            and not self.llm.has_queued_frame(LLMContextFrame)
        )
        text = (message.content or "").strip()
        sent = None
        if text:
            # Default behavior: only the final answer asks to be spoken.
            # This behavior can be adjusted by a transform_output callback.
            sent = await self._emit(
                run, BackendOutput(text=text, is_final=finished, prefers_spoken=finished)
            )
        if finished:
            run.final_text = sent.text if sent else ""
            run.finished.set()

    async def _on_pipeline_error(self, frame: ErrorFrame):
        """End the delegation in progress: the backend cannot answer it.

        A tool handler that raises is the exception. The LLM service reports
        that as ``ErrorCategory.APPLICATION``, settles the call with an error
        result and carries on, so the delegation still has an answer coming.
        """
        run = self._run
        if run is None or frame.category == ErrorCategory.APPLICATION:
            return
        run.error = frame.error
        run.finished.set()

    async def _on_assistant_thought(self, message: AssistantThoughtMessage):
        run = self._run
        text = (message.content or "").strip()
        if run is not None and text:
            await self._emit(run, BackendOutput(text=text, is_thought=True, prefers_spoken=False))

    async def _emit(self, run: "_BackendRun", output: BackendOutput) -> BackendOutput:
        """Send one output as a job update, after any configured transform.

        Returns:
            The output as it was sent, so a caller sees the transformed text.
        """
        if self._transform_output is not None:
            output = await self._transform_output(output)
        if output.text:
            await self.send_job_update(run.job_id, output.to_payload())
        return output


async def delegate_to_backend(
    worker: BaseWorker,
    backend_name: str,
    *,
    request: str,
    timeout_secs: float | None = None,
) -> AsyncIterator[BackendOutput]:
    """Put a request to a :class:`BackendLLMWorker` and yield what it produces.

    Args:
        worker: The worker making the request (for a pipeline processor,
            ``self.pipeline_worker``).
        backend_name: Name of the backend worker.
        request: The text to put to the backend, as its user message.
            :func:`render_transcript_request` composes one from a
            conversation, which is what a frontend hands over when its model
            signals a handoff without wording a request. A frontend whose
            model does word one can send it as it stands::

                async for output in delegate_to_backend(worker, "backend", request=task):
                    ...
        timeout_secs: How long to wait for the backend, including the wait for
            it to become ready.

    Yields:
        Each :class:`BackendOutput` as the backend produces it, the final
        answer last with ``is_final`` set. A delegation that ends without an
        answer yields no final output.

    Raises:
        JobError: If the backend fails, is cancelled, or times out.
    """
    params = JobParams(
        name=BACKEND_JOB_NAME,
        payload={"request": request},
        timeout=timeout_secs,
    )
    async with worker.job(backend_name, params=params) as backend_job:
        async for event in backend_job:
            if event.type != JobEvent.UPDATE or not event.data:
                continue
            if event.data.get("type", OUTPUT_UPDATE_TYPE) != OUTPUT_UPDATE_TYPE:
                continue
            output = BackendOutput.from_payload(event.data)
            if output.text:
                yield output
