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
streams back everything it produces, its final output last, along with the
function calls it makes on the way, for the frontend to report.
:func:`_delegate_to_backend` is the caller side of that contract, an async
iterator over that stream.

A request is text, so how a frontend words one is its own business.
:func:`_render_transcript_request` renders the conversation as a labelled
transcript, which is what a frontend hands over when its model signals a
handoff without wording a request; a frontend whose model does word one sends
that instead.
"""

import asyncio
import inspect
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from loguru import logger

from pipecat.bus.messages import BusJobCancelMessage, BusJobRequestMessage
from pipecat.frames.frames import (
    ErrorFrame,
    ExternalFunctionCallCancelFrame,
    ExternalFunctionCallFrame,
    ExternalFunctionCallInProgressFrame,
    ExternalFunctionCallResultFrame,
    ExternalFunctionCallStartedFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
)
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

#: The ``type`` of a job update that carries a :class:`BackendOutput`.
OUTPUT_UPDATE_TYPE = "output"

#: The ``type`` of a job update that carries a :class:`BackendToolCall`.
TOOL_CALL_UPDATE_TYPE = "tool_call"

#: Appended to the backend LLM's system instruction: its output is relayed to
#: a listener by the frontend, whatever the app's prompt says the backend does.
BACKEND_OUTPUT_INSTRUCTIONS = (
    "Your replies are relayed to a user by a voice assistant. Reply in concise, "
    "conversational plain text it can say aloud: no Markdown, no raw JSON. Never claim "
    "an action completed without a tool result confirming it."
)


@dataclass
class BackendOutput:
    """One piece of output from a backend, on its way to the frontend.

    Parameters:
        text: The text the backend produced.
        is_thought: Whether this is a reasoning summary rather than a response.
        prefers_spoken: Whether the backend would like the user to hear this.
            This is just a hint to the frontend: it may choose to follow it or not.
            ``OpenAILiveLLMService``'s live model takes it into consideration,
            but ultimately decides what to speak (or not) based on the conversation.
    """

    text: str
    is_thought: bool = False
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
            for name in ("is_thought", "prefers_spoken")
            if name in payload
        }
        return cls(text=str(payload.get("text") or ""), **flags)


@dataclass
class BackendToolCall:
    """One phase of a function call the backend made while working, on its way to the frontend.

    The call ran in the backend's own pipeline; the frontend reports it to
    clients, as the :class:`~pipecat.frames.frames.ExternalFunctionCallFrame`
    :meth:`to_frame` builds, and does nothing else with it. The phases mirror
    the pipeline's own function-call frames.

    Parameters:
        phase: ``started``, ``in_progress``, ``result`` or ``cancelled``.
        function_name: Name of the function called.
        tool_call_id: Unique identifier of the call.
        arguments: Arguments passed to the function, once known.
        result: The result, for the ``result`` phase.
        is_final: For the ``result`` phase, whether the result completes the
            call rather than being one of a stream of intermediate results.
    """

    phase: Literal["started", "in_progress", "result", "cancelled"]
    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any] | None = None
    result: Any = None
    is_final: bool = True

    def to_payload(self) -> dict[str, Any]:
        """Render the call phase as a job update payload.

        Returns:
            The payload.
        """
        return {
            "type": TOOL_CALL_UPDATE_TYPE,
            "phase": self.phase,
            "function_name": self.function_name,
            "tool_call_id": self.tool_call_id,
            "arguments": dict(self.arguments) if self.arguments is not None else None,
            "result": self.result,
            "is_final": self.is_final,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BackendToolCall":
        """Rebuild a call phase from a job update payload.

        Args:
            payload: The update payload.

        Returns:
            The call phase.
        """
        return cls(
            phase=payload.get("phase") or "in_progress",
            function_name=str(payload.get("function_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            arguments=payload.get("arguments"),
            result=payload.get("result"),
            is_final=bool(payload.get("is_final", True)),
        )

    def to_frame(self, *, parent_tool_call_id: str | None = None) -> ExternalFunctionCallFrame:
        """Build the frame that reports this phase in the frontend's pipeline.

        Args:
            parent_tool_call_id: The frontend's call this one ran as part of,
                if any.

        Returns:
            The frame for the phase.
        """
        if self.phase == "started":
            return ExternalFunctionCallStartedFrame(
                self.function_name, self.tool_call_id, parent_tool_call_id=parent_tool_call_id
            )
        if self.phase == "in_progress":
            return ExternalFunctionCallInProgressFrame(
                self.function_name,
                self.tool_call_id,
                arguments=self.arguments,
                parent_tool_call_id=parent_tool_call_id,
            )
        if self.phase == "result":
            return ExternalFunctionCallResultFrame(
                self.function_name,
                self.tool_call_id,
                arguments=self.arguments,
                result=self.result,
                is_final=self.is_final,
                parent_tool_call_id=parent_tool_call_id,
            )
        return ExternalFunctionCallCancelFrame(
            self.function_name, self.tool_call_id, parent_tool_call_id=parent_tool_call_id
        )


#: What :func:`_delegate_to_backend` yields: the backend's outputs and the
#: phases of the function calls it made on the way.
@dataclass
class _BackendFinalOutput:
    """The output that settles a delegation, yielded last by :func:`_delegate_to_backend`.

    Parameters:
        output: The final output, from the job's response.
    """

    output: BackendOutput


BackendEvent = BackendOutput | BackendToolCall | _BackendFinalOutput


#: Adjusts a backend output — its text, or whether the user may hear it —
#: before it leaves the worker.
class BackendOutputTransform(Protocol):
    """Shapes one of the backend model's outputs before it is sent.

    Called as ``transform_output(output, is_final=...)``; ``is_final`` is
    true for the final output, which settles the delegation, and false for
    the outputs before it.
    """

    async def __call__(self, output: BackendOutput, *, is_final: bool) -> BackendOutput:
        """Return the output to send in place of ``output``; blank its text to drop it."""
        ...


def _message_text(message: LLMStandardMessage) -> str:
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


#: What :func:`_render_transcript_request` tells the backend to do with a transcript.
_DEFAULT_TRANSCRIPT_INSTRUCTION = "Act on the user's most recent request in the conversation above."


def _render_transcript_request(
    conversation: Sequence[LLMContextMessage],
    *,
    instruction: str = _DEFAULT_TRANSCRIPT_INSTRUCTION,
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
            text = _message_text(message)
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
    final_output: BackendOutput | None = None
    error: str = ""
    finished: asyncio.Event = field(default_factory=asyncio.Event)


class BackendLLMWorker(LLMContextWorker):
    """A worker that runs an LLM service as the backend a frontend delegates to.

    The worker owns the backend's conversation: an ``LLMContext`` plus the
    aggregator pair, so multi-step tool calling works as it does in any
    pipeline. Each delegation arrives as a ``run`` job carrying the request
    text, which is appended to the context as one user message, and runs the
    LLM until it has nothing more to do.

    Everything the backend's model produces reaches the frontend as
    :class:`BackendOutput` updates, continually. :meth:`say` (and its general
    form :meth:`send_output`) lets the app send one output of its own on the
    same channel, e.g. from an ``on_delegation_started`` handler or a tool,
    for a backend that knows its work is slow to tell the user so.

    Event handlers available:

    - on_delegation_started: Called with the request text when a delegation
      arrives, before the model runs.

    Job contract (``@job(name="run")``, one delegation at a time):

    - request payload: ``{"request": str}`` — the text to put to the backend,
      composed by the frontend. What that text says is the application's
      business: :func:`_render_transcript_request` renders the conversation as
      a transcript, which is what a frontend whose model hands off without
      wording a request needs, but a frontend that has a worded request can
      simply send it.
    - updates: a :class:`BackendOutput` payload for every piece of progress —
      reasoning summaries and what the backend says before calling tools —
      and a :class:`BackendToolCall` payload for each phase of each function
      call the backend makes, so the frontend can report them.
    - cancellation: a job the requester cancels interrupts the backend's
      pipeline, so the model stops and the next delegation starts clean.
    - response: the final output as a :class:`BackendOutput` payload, or
      ``{"text": ""}`` if the delegation ended without one. A backend LLM failure fails the job
      with ``JobStatus.ERROR``, so the frontend hears about it as soon as it
      happens.

    Progress arrives as updates and the final output as the response.
    :func:`_delegate_to_backend` wraps the caller side and yields both, the
    final output last.

    Example::

        backend = BackendLLMWorker(
            llm=AnthropicLLMService(api_key=...),
            context=LLMContext(
                [{"role": "system", "content": BACKEND_INSTRUCTIONS}],
                tools=[get_weather],
            ),
        )

        @backend.event_handler("on_delegation_started")
        async def on_delegation_started(backend, request):
            await backend.say("Let me look into that, this takes a moment.")
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
            transform_output: Called with each :class:`BackendOutput` the
                model produces before it is sent, as
                ``transform_output(output, is_final=...)``, to adjust its
                text or whether the user may hear it; an output with no text
                left is not sent. Without one, only the final output asks to be
                spoken: a frontend filling the wait is usually mid-sentence
                when progress arrives, and speaking it talks over them.
                Outputs the app sends itself are not passed through it
                unless the call asks.
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
        if transform_output is not None:
            _validate_transform_signature(transform_output)
        self._transform_output = transform_output
        self.llm.append_system_instruction(BACKEND_OUTPUT_INSTRUCTIONS)
        self._register_event_handler("on_delegation_started")

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

        # The backend's function calls are relayed as they pass the assistant
        # aggregator, which every phase of a call reaches.
        @self.assistant_aggregator.event_handler("on_before_process_frame")
        async def on_before_aggregator_frame(aggregator, frame: Frame):
            await self._on_function_call_frame(frame)

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
        await self._call_event_handler("on_delegation_started", request)
        await self.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": request}], run_llm=True)
        )
        try:
            await run.finished.wait()
        finally:
            self._run = None
        # An error fails the job only when it left the backend with nothing to say.
        if run.error and run.final_output is None:
            logger.warning(f"Worker '{self.name}': job {message.job_id} failed: {run.error}")
            await self.send_job_response(message.job_id, {"text": ""}, status=JobStatus.ERROR)
            return
        await self.send_job_response(
            message.job_id,
            run.final_output.to_payload() if run.final_output else {"text": ""},
        )

    async def say(self, text: str, *, apply_transform_output: bool = False) -> None:
        """Send one spoken line to the frontend, on the delegation in progress.

        The line is a :class:`BackendOutput` flagged ``prefers_spoken``, so a
        frontend following the flag says it right away while the backend keeps
        working. For anything else, :meth:`send_output`.

        Args:
            text: What the frontend should say.
            apply_transform_output: Whether to run the line through
                ``transform_output`` as the model's outputs are; see
                :meth:`send_output`.
        """
        await self.send_output(
            BackendOutput(text=text, prefers_spoken=True),
            apply_transform_output=apply_transform_output,
        )

    async def send_output(
        self, output: BackendOutput, *, apply_transform_output: bool = False
    ) -> None:
        """Send one output of the app's own to the frontend, on the delegation in progress.

        It reaches the frontend as the model's outputs do, but as given unless
        asked: ``transform_output`` is not applied to it by default. Outside a
        delegation there is nowhere for it to go, and it is dropped with a
        warning.

        Args:
            output: The output.
            apply_transform_output: Whether to run the output through
                ``transform_output`` as the model's outputs are.
        """
        run = self._run
        if run is None:
            logger.warning(f"Worker '{self.name}': no delegation in progress to send output on")
            return
        # Past the transform by default, so an app can silence the model's
        # progress with a transform_output that blanks it and still send
        # progress of its own, such as from a tool the model calls to talk to
        # the user.
        await self._emit(run, output, apply_transform_output=apply_transform_output)

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
        # Default behavior: only the final output asks to be spoken.
        # This behavior can be adjusted by a transform_output callback.
        try:
            if finished:
                # The final output is the job's response, not an update.
                final = BackendOutput(text=text, prefers_spoken=True) if text else None
                if final is not None:
                    final = await self._shape(final, is_final=True)
                run.final_output = final if final and final.text else None
                run.finished.set()
            elif text:
                await self._emit(run, BackendOutput(text=text, prefers_spoken=False))
        except Exception as e:
            # A transform that raises would otherwise leave the delegation
            # waiting out the caller's timeout; fail the job instead.
            logger.error(f"Worker '{self.name}': transform_output failed: {e}")
            run.error = f"transform_output failed: {e}"
            run.finished.set()

    async def _on_function_call_frame(self, frame: Frame):
        """Relay a phase of one of the backend's own function calls as a job update."""
        run = self._run
        if run is None:
            return
        calls: list[BackendToolCall] = []
        if isinstance(frame, FunctionCallsStartedFrame):
            calls = [
                BackendToolCall("started", call.function_name, call.tool_call_id)
                for call in frame.function_calls
            ]
        elif isinstance(frame, FunctionCallInProgressFrame):
            calls = [
                BackendToolCall(
                    "in_progress",
                    frame.function_name,
                    frame.tool_call_id,
                    arguments=frame.arguments,
                )
            ]
        elif isinstance(frame, FunctionCallResultFrame):
            calls = [
                BackendToolCall(
                    "result",
                    frame.function_name,
                    frame.tool_call_id,
                    arguments=frame.arguments,
                    result=frame.result,
                    is_final=frame.properties is None or frame.properties.is_final,
                )
            ]
        elif isinstance(frame, FunctionCallCancelFrame):
            calls = [BackendToolCall("cancelled", frame.function_name, frame.tool_call_id)]
        for call in calls:
            await self.send_job_update(run.job_id, call.to_payload())

    async def on_job_cancelled(self, message: BusJobCancelMessage) -> None:
        """Stop the model: the requester no longer wants its output.

        The delegation's handler is already cancelled by the time this runs;
        interrupting the pipeline stops the run in flight and any tool call
        it is waiting on, so nothing of it reaches the next delegation.
        """
        await self.queue_frame(InterruptionFrame())

    async def _on_pipeline_error(self, frame: ErrorFrame):
        """End the delegation in progress: the backend cannot finish it.

        A tool handler that raises is the exception. The LLM service reports
        that as ``ErrorCategory.APPLICATION``, settles the call with an error
        result and carries on, so the delegation still has a final output coming.
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

    async def _shape(self, output: BackendOutput, *, is_final: bool) -> BackendOutput:
        """Run an output through ``transform_output``, if there is one."""
        if self._transform_output is None:
            return output
        return await self._transform_output(output, is_final=is_final)

    async def _emit(
        self, run: "_BackendRun", output: BackendOutput, *, apply_transform_output: bool = True
    ) -> BackendOutput:
        """Send one output as a job update, after any configured transform.

        Returns:
            The output as it was sent, so a caller sees the transformed text.
        """
        if apply_transform_output:
            output = await self._shape(output, is_final=False)
        if output.text:
            await self.send_job_update(run.job_id, output.to_payload())
        return output


def _validate_transform_signature(transform: BackendOutputTransform) -> None:
    """Reject a ``transform_output`` that cannot take ``is_final`` as a keyword.

    A transform written before 1.12.0 took the output alone; this turns that
    into an error at construction rather than on the first delegation.
    """
    try:
        parameters = inspect.signature(transform).parameters
    except (TypeError, ValueError):
        return
    if "is_final" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return
    raise TypeError(
        "transform_output must take is_final as a keyword: "
        "async def transform_output(output: BackendOutput, *, is_final: bool) -> BackendOutput"
    )


async def _delegate_to_backend(
    worker: BaseWorker,
    backend_name: str,
    *,
    request: str,
    timeout_secs: float | None = None,
) -> AsyncGenerator[BackendEvent, None]:
    """Put a request to a :class:`BackendLLMWorker` and yield what it produces.

    Args:
        worker: The worker making the request (for a pipeline processor,
            ``self.pipeline_worker``).
        backend_name: Name of the backend worker.
        request: The text to put to the backend, as its user message.
            :func:`_render_transcript_request` composes one from a
            conversation, which is what a frontend hands over when its model
            signals a handoff without wording a request. A frontend whose
            model does word one can send it as it stands::

                async for output in _delegate_to_backend(worker, "backend", request=request):
                    ...
        timeout_secs: How long to wait for the backend, including the wait for
            it to become ready.

    Yields:
        Each :class:`BackendOutput` as the backend produces it, the final
        output last as a :class:`_BackendFinalOutput`, and each
        :class:`BackendToolCall` phase as the backend's calls run. The final
        output always comes; an output's text may be empty.

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
            update_type = event.data.get("type", OUTPUT_UPDATE_TYPE)
            if update_type == OUTPUT_UPDATE_TYPE:
                yield BackendOutput.from_payload(event.data)
            elif update_type == TOOL_CALL_UPDATE_TYPE:
                yield BackendToolCall.from_payload(event.data)
        yield _BackendFinalOutput(BackendOutput.from_payload(backend_job.response))
