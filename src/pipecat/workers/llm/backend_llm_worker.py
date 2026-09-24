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
and multi-step tool calling, to do that work.

The frontend and the backend exchange messages, not calls. A frontend
*attaches* to the worker once, for as long as it lives, and from then on
hears everything the backend produces as a stream: each of its model's turns
and reasoning summaries as a :class:`BackendOutput`, each phase of the
function calls it makes as a :class:`BackendToolCall`, and when it has run
out of things to do, that it is idle. Requests go the other way as messages
appended to the backend's conversation, each answered at once; a message that
arrives while the backend is working joins that work, and the backend's model
sorts out what it means. :class:`_BackendSession` is the caller side of that
contract.

A request is text, so how a frontend words one is its own business.
:func:`_render_transcript_request` renders the conversation as a labelled
transcript, which is what a frontend hands over when its model signals a
handoff without wording a request; a frontend whose model does word one sends
that instead.

The ``run`` job, a bounded delegation that ends with a final output, is the
contract :class:`~pipecat.services.openai.live.llm.OpenAILiveLLMService`'s
client delegation still drives; :func:`_delegate_to_backend` is its caller
side.
"""

import asyncio
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
from pipecat.pipeline.job_context import (
    JobError,
    JobEvent,
    JobGroup,
    JobGroupError,
    JobGroupParams,
    JobParams,
    JobStatus,
)
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

#: Name of the job that attaches a frontend to a :class:`BackendLLMWorker`.
ATTACH_JOB_NAME = "attach"

#: Name of the job that puts a message to an attached :class:`BackendLLMWorker`.
MESSAGE_JOB_NAME = "message"

#: Name of the job that stops what an attached :class:`BackendLLMWorker` is doing.
CANCEL_JOB_NAME = "cancel"

#: Name of the bounded-delegation job a :class:`BackendLLMWorker` handles.
BACKEND_JOB_NAME = "run"

#: The ``type`` of a job update that carries a :class:`BackendOutput`.
OUTPUT_UPDATE_TYPE = "output"

#: The ``type`` of a job update that carries a :class:`BackendToolCall`.
TOOL_CALL_UPDATE_TYPE = "tool_call"

#: The ``type`` of the update that opens an ``attach`` job's stream.
ATTACHED_UPDATE_TYPE = "attached"

#: The ``type`` of the update that says the backend has nothing left to do.
IDLE_UPDATE_TYPE = "idle"

#: The ``type`` of the update that says the backend could not go on.
ERROR_UPDATE_TYPE = "error"

#: Appended to the backend LLM's system instruction: where its input comes
#: from and its output goes, whatever the app's prompt says the backend does.
BACKEND_OUTPUT_INSTRUCTIONS = (
    "You are the backend of a voice assistant. What you receive comes from the assistant: "
    "the conversation it is having with the user, or a request it wrote for you. What you "
    "write goes to the assistant, which decides what the user hears and says it in its own "
    "words. Write plain text it can speak from: no Markdown, no raw JSON. Keep progress "
    "notes to a sentence; give results in full. A new message may arrive while you are "
    "working. It may add work, change it, or cancel some or all of it: follow the latest "
    "instructions, cancel tools whose results are no longer wanted, and do not repeat work "
    "already done. If your work is cancelled, do not announce it; the assistant already "
    "has. Ask the assistant a question only when you cannot proceed without an answer."
)

#: Appended to the backend's conversation when the frontend cancels its work.
CANCELLED_NOTE = (
    "The user cancelled the work in progress. Do nothing further on it unless asked again."
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


@dataclass
class BackendIdle:
    """The backend has nothing left to do: no run outstanding, no call in flight."""


@dataclass
class BackendError:
    """The backend could not go on with what it was doing.

    Parameters:
        error: What went wrong.
    """

    error: str


@dataclass
class _BackendFinalOutput:
    """The output that settles a ``run`` delegation, yielded last by :func:`_delegate_to_backend`.

    Parameters:
        output: The final output, from the job's response.
    """

    output: BackendOutput


#: What a :class:`_BackendSession` yields.
BackendEvent = BackendOutput | BackendToolCall | BackendIdle | BackendError

#: What :func:`_delegate_to_backend` yields.
BackendRunEvent = BackendOutput | BackendToolCall | _BackendFinalOutput


class BackendOutputTransform(Protocol):
    """Shapes one of the backend model's outputs before it is sent."""

    async def __call__(self, output: BackendOutput) -> BackendOutput | None:
        """Return the output to send in place of ``output``, or ``None`` to send nothing."""
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
    """A ``run`` job in progress: one delegation and the LLM runs it takes."""

    job_id: str
    final_output: BackendOutput | None = None
    error: str = ""
    finished: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _AttachedFrontend:
    """The frontend attached to the worker, for as long as its ``attach`` job lives."""

    job_id: str
    detached: asyncio.Event = field(default_factory=asyncio.Event)


class BackendLLMWorker(LLMContextWorker):
    """A worker that runs an LLM service as the backend a frontend delegates to.

    The worker owns the backend's conversation: an ``LLMContext`` plus the
    aggregator pair, so multi-step tool calling works as it does in any
    pipeline. Requests are appended to that conversation as user messages
    and run the model; whatever it produces reaches the frontend as it is
    produced, over one of two contracts, never both at once:

    - **Attached** (``attach``, ``message``, ``cancel``): what
      :class:`~pipecat.pipeline.llm_with_backend.LLMWithBackend` drives. The
      frontend attaches once and hears every output for as long as it stays
      attached; each message it sends is answered at once and joins whatever
      the backend is doing. :class:`_BackendSession` is the caller side.
    - **Run** (``run``): a bounded delegation that streams its outputs and ends
      with a final output, which
      :class:`~pipecat.services.openai.live.llm.OpenAILiveLLMService`'s client
      delegation drives one at a time. :func:`_delegate_to_backend` is the
      caller side.

    Job contract, attached:

    - ``attach``: no payload. Its first update is ``{"type": "attached",
      "capabilities": {...}}``; after that, a :class:`BackendOutput` payload
      for each of the model's turns and reasoning summaries and each output the
      app sends with :meth:`send_output`, a :class:`BackendToolCall` payload for
      each phase of each function call the backend makes, ``{"type": "idle"}``
      when the backend has nothing left to do, and ``{"type": "error", "error":
      ...}`` when it could not go on. The job lives until the requester cancels
      it, which stops the backend's work. Refused while another frontend is
      attached or a ``run`` is in progress.
    - ``message``: ``{"request": str}``, appended as a user message; responds at
      once with ``{"backend": "idle" | "working"}``, what the backend was doing
      when the message arrived. A message that arrives mid-run is taken up after
      the current step, with everything that came before it in view.
    - ``cancel``: ``{"reason": str}``. Interrupts the backend's pipeline, cancels
      every function call in flight, async ones included, and appends
      :data:`CANCELLED_NOTE`; responds with ``{"cancelled": bool}``, whether
      there was anything to stop.

    Job contract, ``run`` (one delegation at a time): request payload
    ``{"request": str}``; updates as above, without ``idle``; the response is
    the final output as a :class:`BackendOutput` payload, or ``{"text": ""}``
    if the delegation ended without one. A backend LLM failure fails the job
    with ``JobStatus.ERROR``. Cancelling the job interrupts the backend's
    pipeline, so the next delegation starts clean.

    Whether an output asks to be spoken is decided per output: a turn that
    ended by calling tools is narration and does not, a turn that did not
    call tools does, a reasoning summary does not. ``transform_output`` can
    change that, or the text, or drop the output.

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
            name: Unique name for this worker on the bus. Auto-generated when
                omitted; give one when the backend runs in another process,
                where the frontend addresses it by name.
            transform_output: Called with each :class:`BackendOutput` the
                model produces before it is sent, to adjust its text or
                whether the user may hear it, or to return ``None`` and send
                nothing. Outputs the app sends itself are not passed through
                it unless the call asks.
            user_params: Optional parameters for the user aggregator. Defaults
                to external turn strategies: the backend has no audio, so the
                default VAD and turn-analysis strategies (and the model the
                latter loads) have nothing to do here.
            assistant_params: Optional parameters for the assistant aggregator.
        """
        if user_params is None:
            user_params = LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies())
        super().__init__(
            name,
            llm=llm,
            active=True,
            context=context,
            user_params=user_params,
            assistant_params=assistant_params,
        )
        self._run: _BackendRun | None = None
        self._attached: _AttachedFrontend | None = None
        self._transform_output = transform_output
        self.llm.append_system_instruction(BACKEND_OUTPUT_INSTRUCTIONS)

        # Whether the backend is working is read off its pipeline: requests
        # queued but not yet taken up, LLM runs picked up but not yet ended,
        # function calls in flight, and context frames queued for the LLM. A
        # request takes one or more runs: the first for the request itself,
        # then one per round of tool results (the assistant aggregator pushes
        # an LLMContextFrame back to the LLM after each). Runs are counted as
        # the LLM picks them up and as their turns end; an interruption
        # cancels whatever was outstanding.
        self._requests_pending = 0
        self._runs_requested = 0
        self._runs_completed = 0
        # Whether the turn in progress has called tools, which decides whether
        # its text is narration or something to say. Set when the calls start
        # and cleared when the turn ends: the calls-started frame is a system
        # frame, so it reaches the aggregator ahead of the queued frames that
        # open the turn it belongs to.
        self._turn_made_calls = False

        @self.llm.event_handler("on_before_process_frame")
        async def on_before_llm_frame(llm, frame: Frame):
            if isinstance(frame, LLMContextFrame):
                self._runs_requested += 1

        @self.user_aggregator.event_handler("on_before_process_frame")
        async def on_before_user_aggregator_frame(aggregator, frame: Frame):
            if isinstance(frame, LLMMessagesAppendFrame) and self._requests_pending:
                self._requests_pending -= 1

        # The backend's function calls are relayed as they pass the assistant
        # aggregator, which every phase of a call reaches.
        @self.assistant_aggregator.event_handler("on_before_process_frame")
        async def on_before_assistant_aggregator_frame(aggregator, frame: Frame):
            if isinstance(frame, InterruptionFrame):
                self._runs_requested = self._runs_completed = 0
                self._turn_made_calls = False
            elif isinstance(frame, FunctionCallsStartedFrame):
                self._turn_made_calls = True
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

    @property
    def working(self) -> bool:
        """Whether the backend has work in hand: a request or run outstanding, or a call in flight."""
        return (
            self._requests_pending > 0
            or self._runs_requested > self._runs_completed
            or self.assistant_aggregator.has_function_calls_in_progress
            or self.llm.has_queued_frame(LLMContextFrame)
        )

    @property
    def capabilities(self) -> dict[str, bool]:
        """What an attached frontend can count on; all of it, for a backend that is an LLM."""
        return {"steering": True, "cancellation": True, "progress": True}

    # -----------------------------------------------------------------------
    # The attached contract
    # -----------------------------------------------------------------------

    @job(name=ATTACH_JOB_NAME)
    async def attach_frontend(self, message: BusJobRequestMessage):
        """Attach a frontend: stream everything the backend produces until the job is cancelled.

        Args:
            message: The job request.
        """
        if self._attached is not None:
            await self._refuse(message.job_id, "a frontend is already attached")
            return
        if self._run is not None:
            await self._refuse(message.job_id, "a run job is in progress")
            return
        attached = self._attached = _AttachedFrontend(job_id=message.job_id)
        await self.send_job_update(
            message.job_id, {"type": ATTACHED_UPDATE_TYPE, "capabilities": self.capabilities}
        )
        try:
            # Ends only by cancellation, which is how the frontend detaches.
            await attached.detached.wait()
        finally:
            if self._attached is attached:
                self._attached = None

    @job(name=MESSAGE_JOB_NAME)
    async def handle_message(self, message: BusJobRequestMessage):
        """Put a message to the backend, and say what it was doing when it arrived.

        Args:
            message: The job request; see the class docstring for the payload.
        """
        if self._attached is None:
            await self._refuse(message.job_id, "no frontend is attached; open an attach job first")
            return
        request = str((message.payload or {}).get("request") or "").strip()
        if not request:
            await self._refuse(message.job_id, "the message has no request")
            return
        status = "working" if self.working else "idle"
        await self._queue_request(request)
        await self.send_job_response(message.job_id, {"backend": status})

    @job(name=CANCEL_JOB_NAME)
    async def handle_cancel(self, message: BusJobRequestMessage):
        """Stop the backend's work, and say whether there was any.

        Args:
            message: The job request; see the class docstring for the payload.
        """
        if self._attached is None:
            await self._refuse(message.job_id, "no frontend is attached")
            return
        reason = str((message.payload or {}).get("reason") or "cancelled by the frontend")
        was_working = self.working
        await self._stop_work(reason)
        note = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": CANCELLED_NOTE}], run_llm=False
        )
        note.interruptible = False
        await self.queue_frame(note)
        await self.send_job_response(message.job_id, {"cancelled": was_working})

    async def _refuse(self, job_id: str, reason: str) -> None:
        logger.warning(f"Worker '{self.name}': refusing job {job_id}: {reason}")
        await self.send_job_response(job_id, {"error": reason}, status=JobStatus.ERROR)

    async def _queue_request(self, request: str) -> None:
        """Append a request to the backend's conversation and run the model on it.

        The frame is uninterruptible, so a request queued just ahead of a
        cancellation survives the interruption the cancellation broadcasts.
        """
        self._requests_pending += 1
        frame = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": request}], run_llm=True
        )
        frame.interruptible = False
        await self.queue_frame(frame)

    async def _stop_work(self, reason: str) -> None:
        """Interrupt the pipeline and cancel every function call in flight.

        An interruption stops the run in progress and the calls that opt into
        cancellation on interruption; the async ones, which an interruption
        leaves running by design, are cancelled on their own.
        """
        await self.queue_frame(InterruptionFrame())
        await self.llm.cancel_function_calls(reason=reason)

    # -----------------------------------------------------------------------
    # The run contract
    # -----------------------------------------------------------------------

    @job(name=BACKEND_JOB_NAME, sequential=True)
    async def run_delegation(self, message: BusJobRequestMessage):
        """Run one delegation to completion, streaming what the backend produces as updates.

        Args:
            message: The job request; see the class docstring for the payload.
        """
        if self._attached is not None:
            await self._refuse(message.job_id, "a frontend is attached; use a message job")
            return
        request = str((message.payload or {}).get("request") or "").strip()
        if not request:
            logger.warning(f"Worker '{self.name}': job {message.job_id} has no request")
            await self.send_job_response(message.job_id, {"text": ""})
            return

        run = self._run = _BackendRun(job_id=message.job_id)
        await self._queue_request(request)
        try:
            await run.finished.wait()
        finally:
            self._run = None
        # An error fails the job only when no final output with text came of the run.
        if run.error and not (run.final_output and run.final_output.text):
            logger.warning(f"Worker '{self.name}': job {message.job_id} failed: {run.error}")
            await self.send_job_response(message.job_id, {"text": ""}, status=JobStatus.ERROR)
            return
        await self.send_job_response(
            message.job_id,
            run.final_output.to_payload()
            if run.final_output
            else {"text": "", "prefers_spoken": False},
        )

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    async def send_output(
        self, output: BackendOutput, *, apply_transform_output: bool = False
    ) -> None:
        """Send one output of the app's own to the frontend.

        It reaches the frontend as the model's outputs do, but as given unless
        asked: ``transform_output`` is not applied to it by default. With no
        frontend attached and no run in progress there is nowhere for it to
        go, and it is dropped with a warning.

        Args:
            output: The output.
            apply_transform_output: Whether to run the output through
                ``transform_output`` as the model's outputs are.
        """
        if self._attached is None and self._run is None:
            logger.warning(f"Worker '{self.name}': no frontend to send output to")
            return
        # Past the transform by default, so an app can silence the model's
        # progress with a transform_output that drops it and still send
        # progress of its own, such as from a tool the model calls to talk to
        # the user.
        await self._emit(output, apply_transform_output=apply_transform_output)

    async def _on_assistant_turn_stopped(self, message: AssistantTurnStoppedMessage):
        if message.interrupted:
            # What an interrupted turn produced is moot, and the interruption
            # already zeroed the runs outstanding.
            return
        self._runs_completed += 1
        text = (message.content or "").strip()
        made_calls, self._turn_made_calls = self._turn_made_calls, False
        if self._run is not None:
            await self._on_run_turn_stopped(self._run, text)
            return
        if self._attached is None:
            return
        try:
            if text:
                await self._emit(BackendOutput(text=text, prefers_spoken=not made_calls))
        except Exception as e:
            logger.error(f"Worker '{self.name}': transform_output failed: {e}")
            await self._send_update({"type": ERROR_UPDATE_TYPE, "error": str(e)})
        if not self.working:
            await self._send_update({"type": IDLE_UPDATE_TYPE})

    async def _on_run_turn_stopped(self, run: _BackendRun, text: str):
        # The delegation is finished when no further run is on the way. The
        # check assumes a settled tool call leaves the LLM something more to
        # do: a tool that returns no result, or passes run_llm=False, runs
        # nothing further, so the delegation waits out the caller's timeout.
        finished = not self.working
        try:
            if finished:
                # The final output is the job's response, not an update.
                run.final_output = await self._shape(BackendOutput(text=text, prefers_spoken=True))
                run.finished.set()
            elif text:
                await self._emit(BackendOutput(text=text, prefers_spoken=False))
        except Exception as e:
            # A transform that raises would otherwise leave the delegation
            # waiting out the caller's timeout; fail the job instead.
            logger.error(f"Worker '{self.name}': transform_output failed: {e}")
            run.error = f"transform_output failed: {e}"
            run.finished.set()

    async def _on_function_call_frame(self, frame: Frame):
        """Relay a phase of one of the backend's own function calls as a job update."""
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
            await self._send_update(call.to_payload())

    async def on_job_cancelled(self, message: BusJobCancelMessage) -> None:
        """Stop the model: the requester no longer wants its output.

        The job's handler is already cancelled by the time this runs. For an
        ``attach`` job, the frontend is gone; for a ``run`` job, the delegation
        is abandoned. Either way the work in flight stops, so nothing of it
        reaches the next requester.
        """
        if self._attached is not None and message.job_id == self._attached.job_id:
            self._attached = None
        await self._stop_work("cancelled by the requester")

    async def _on_pipeline_error(self, frame: ErrorFrame):
        """Tell the frontend the backend cannot go on with what it was doing.

        A tool handler that raises is the exception. The LLM service reports
        that as ``ErrorCategory.APPLICATION``, settles the call with an error
        result and carries on, so the model still has something coming.
        """
        if frame.category == ErrorCategory.APPLICATION:
            return
        if self._run is not None:
            self._run.error = frame.error
            self._run.finished.set()
        elif self._attached is not None:
            await self._send_update({"type": ERROR_UPDATE_TYPE, "error": frame.error})

    async def _on_assistant_thought(self, message: AssistantThoughtMessage):
        text = (message.content or "").strip()
        if text and (self._attached is not None or self._run is not None):
            await self._emit(BackendOutput(text=text, is_thought=True, prefers_spoken=False))

    async def _shape(self, output: BackendOutput) -> BackendOutput | None:
        """Run an output through ``transform_output``, if there is one."""
        if self._transform_output is None:
            return output
        return await self._transform_output(output)

    async def _emit(self, output: BackendOutput, *, apply_transform_output: bool = True) -> None:
        """Send one output as a job update, after any configured transform."""
        shaped = await self._shape(output) if apply_transform_output else output
        if shaped is not None:
            await self._send_update(shaped.to_payload())

    async def _send_update(self, payload: dict[str, Any]) -> None:
        """Send a job update to whoever is listening: the attached frontend, or the run's requester."""
        if self._attached is not None:
            await self.send_job_update(self._attached.job_id, payload)
        elif self._run is not None:
            await self.send_job_update(self._run.job_id, payload)


def _event_from_payload(payload: dict[str, Any]) -> BackendEvent | None:
    """Turn an ``attach`` stream update into the event it carries, if it carries one."""
    update_type = payload.get("type", OUTPUT_UPDATE_TYPE)
    if update_type == OUTPUT_UPDATE_TYPE:
        return BackendOutput.from_payload(payload)
    if update_type == TOOL_CALL_UPDATE_TYPE:
        return BackendToolCall.from_payload(payload)
    if update_type == IDLE_UPDATE_TYPE:
        return BackendIdle()
    if update_type == ERROR_UPDATE_TYPE:
        return BackendError(error=str(payload.get("error") or ""))
    return None


class _BackendSession:
    """A frontend's side of the attached contract: one worker, attached for the session's life.

    Opening the session sends the ``attach`` job; closing it cancels the job,
    which stops the backend's work. In between, the session is an async
    iterator over what the backend produces, and :meth:`send` and
    :meth:`cancel` put messages to it. The iteration ends when the backend
    goes away, with a :class:`JobError` if it failed.

    Example::

        async with _BackendSession(worker, "backend") as session:
            await session.send(request)
            async for event in session:
                ...
    """

    def __init__(self, worker: BaseWorker, backend_name: str, *, timeout_secs: float | None = 30):
        """Initialize the session.

        Args:
            worker: The worker attaching (for a pipeline processor,
                ``self.pipeline_worker``).
            backend_name: Name of the backend worker.
            timeout_secs: How long a message or cancellation may take to be
                acknowledged, including the wait for the backend to become
                ready. The attachment itself has no timeout.
        """
        self._worker = worker
        self._backend_name = backend_name
        self._timeout_secs = timeout_secs
        self._group: JobGroup | None = None
        self._capabilities: dict[str, bool] = {}

    @property
    def backend_name(self) -> str:
        """Name of the backend worker."""
        return self._backend_name

    @property
    def capabilities(self) -> dict[str, bool]:
        """What the backend said it can do, once its stream has opened."""
        return self._capabilities

    async def __aenter__(self) -> "_BackendSession":
        self._group = await self._worker.create_job_group_and_request_job(
            [self._backend_name], params=JobGroupParams(name=ATTACH_JOB_NAME)
        )
        self._group.event_queue = asyncio.Queue()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        """Detach: cancel the ``attach`` job, which stops the backend's work."""
        if self._group and self._group.job_id in self._worker.job_groups:
            await asyncio.shield(
                self._worker.cancel_job_group(self._group.job_id, reason="frontend detached")
            )

    def __aiter__(self):
        return self

    async def __anext__(self) -> BackendEvent:
        assert self._group is not None and self._group.event_queue is not None, "not open"
        while True:
            event = await self._group.event_queue.get()
            if event is None:
                try:
                    await self._group.wait()
                except JobGroupError as e:
                    raise JobError(
                        _with_reason(str(e), self._group.responses.get(self._backend_name))
                    ) from e
                raise StopAsyncIteration
            if event.type != JobEvent.UPDATE or not event.data:
                continue
            if event.data.get("type") == ATTACHED_UPDATE_TYPE:
                self._capabilities = dict(event.data.get("capabilities") or {})
                continue
            parsed = _event_from_payload(event.data)
            if parsed is not None:
                return parsed

    async def send(self, request: str) -> str:
        """Put a message to the backend.

        Args:
            request: The text to append to the backend's conversation.

        Returns:
            What the backend was doing when the message arrived: ``"idle"``
            or ``"working"``.

        Raises:
            JobError: If the backend refused the message or did not answer in time.
        """
        params = JobParams(
            name=MESSAGE_JOB_NAME, payload={"request": request}, timeout=self._timeout_secs
        )
        response = await self._ask(params)
        return str(response.get("backend") or "idle")

    async def cancel(self, reason: str) -> bool:
        """Stop the backend's work.

        Args:
            reason: Why, for the backend's logs.

        Returns:
            Whether there was work to stop.

        Raises:
            JobError: If the backend refused or did not answer in time.
        """
        params = JobParams(
            name=CANCEL_JOB_NAME, payload={"reason": reason}, timeout=self._timeout_secs
        )
        response = await self._ask(params)
        return bool(response.get("cancelled"))

    async def _ask(self, params: JobParams) -> dict:
        """Run one short job on the backend and return its response.

        Raises:
            JobError: If the backend refused, naming its reason, or did not answer in time.
        """
        try:
            async with self._worker.job(self._backend_name, params=params) as short_job:
                pass
        except JobError as e:
            raise JobError(_with_reason(str(e), short_job.response)) from e
        return short_job.response


def _with_reason(error: str, response: dict | None) -> str:
    """The job error with the backend's own reason for it, when its response gave one."""
    reason = (response or {}).get("error")
    return f"{error}: {reason}" if reason else error


async def _delegate_to_backend(
    worker: BaseWorker,
    backend_name: str,
    *,
    request: str,
    timeout_secs: float | None = None,
) -> AsyncGenerator[BackendRunEvent, None]:
    """Put a ``run`` delegation to a :class:`BackendLLMWorker` and yield what it produces.

    Args:
        worker: The worker making the request (for a pipeline processor,
            ``self.pipeline_worker``).
        backend_name: Name of the backend worker.
        request: The text to put to the backend, as its user message.
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
