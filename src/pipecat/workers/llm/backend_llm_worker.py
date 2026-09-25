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

The frontend and the backend exchange messages. A frontend
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
that instead. :class:`~pipecat.services.openai.live.llm.OpenAILiveLLMService`'s
client delegation and :class:`~pipecat.pipeline.llm_with_backend.LLMWithBackend`
both drive the worker this way.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
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
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
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
    standard_message_text,
)
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantThoughtMessage,
    AssistantTurnStoppedMessage,
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import FunctionCallParams, LLMService
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
#: Name of the built-in tool the backend's model calls to tell the user something
#: while it goes on working.
REPORT_TOOL_NAME = "report_result"

BACKEND_OUTPUT_INSTRUCTIONS = (
    "You are the backend of a voice assistant. What you receive comes from the assistant: "
    "the conversation it is having with the user, or a request it wrote for you. What "
    "reaches the user depends on how you write it. A message with no tool calls is told to "
    "the user: write one to report a result, or to ask a question you cannot proceed "
    "without. Text beside tool calls is not told to the user; it is notes on what you are "
    "doing, and may be left out. To tell the user something while you go on working, call "
    f"{REPORT_TOOL_NAME} with it, in the same message as the tool calls that go on, and do "
    "not repeat it afterwards. Report the result of each request as soon as you have it, and "
    "never let it wait until other work is done. Write plain text the assistant can speak "
    "from: no Markdown, no raw JSON. When you have several requests, do the newest first "
    "unless it depends on an earlier one. A new message may arrive while you are working. It "
    "may add work, change it, or cancel some or all of it: follow the latest instructions, "
    "cancel tools whose results are no longer wanted, and do not repeat work already done. "
    "If your work is cancelled, do not announce it; the assistant already has."
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


#: What a :class:`_BackendSession` yields.
BackendEvent = BackendOutput | BackendToolCall | BackendIdle | BackendError


class BackendOutputTransform(Protocol):
    """Shapes one of the backend model's outputs before it is sent."""

    async def __call__(self, output: BackendOutput) -> BackendOutput | None:
        """Return the output to send in place of ``output``, or ``None`` to send nothing."""
        ...


#: What :func:`_render_transcript_request` tells the backend to do with a transcript.
_DEFAULT_TRANSCRIPT_INSTRUCTION = (
    "Act on the user's most recent request in the conversation above, and report its result "
    "as soon as you have it, before going on with other work."
)

#: Appended to a request the frontend worded itself, so the backend reports it
#: on its own rather than with whatever else it is doing.
_REPORT_INSTRUCTION = (
    "Report the result of this request as soon as you have it, before going on with other work."
)


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
            text = standard_message_text(message)
            if role in ("user", "assistant") and text:
                lines.append(f"{str(role).upper()}: {text}")
            else:
                logger.debug(f"Skipping delegated message with no transcript form: role={role!r}")
        lines.append("")
    lines.append(instruction)
    return "\n".join(lines)


@dataclass
class _AttachedFrontend:
    """The frontend attached to the worker, for as long as its ``attach`` job lives."""

    job_id: str
    detached: asyncio.Event = field(default_factory=asyncio.Event)


class BackendLLMWorker(LLMContextWorker):
    """A worker that runs an LLM service as the backend a frontend delegates to.

    The worker owns the backend's conversation: an ``LLMContext`` plus the
    aggregator pair, so multi-step tool calling works as it does in any
    pipeline. A frontend attaches once, for as long as it lives, and from then
    on hears everything the backend produces; requests it sends are appended
    to that conversation as user messages and run the model.
    :class:`_BackendSession` is the caller side.

    Job contract:

    - ``attach``: no payload. Its first update is ``{"type": "attached",
      "capabilities": {...}}``; after that, a :class:`BackendOutput` payload
      for each of the model's turns and reasoning summaries and each output the
      app sends with :meth:`send_output`, a :class:`BackendToolCall` payload for
      each phase of each function call the backend makes, ``{"type": "idle"}``
      when the backend has nothing left to do, and ``{"type": "error", "error":
      ...}`` when it could not go on. The job lives until the requester cancels
      it, which stops the backend's work. Refused while another frontend is
      attached.
    - ``message``: ``{"request": str}``, appended as a user message; responds at
      once with ``{"backend": "idle" | "working"}``, what the backend was doing
      when the message arrived. A message that arrives mid-run is taken up after
      the current step, with everything that came before it in view.
    - ``cancel``: ``{"reason": str}``. Interrupts the backend's pipeline, cancels
      every function call in flight, async ones included, and appends
      :data:`CANCELLED_NOTE`; responds with ``{"cancelled": bool}``, whether
      there was anything to stop.

    What the model writes in a turn with no tool calls asks to be spoken: a
    result, or a question it cannot proceed without. What it writes beside
    tool calls does not, since that is notes on the work in progress; nor
    does a reasoning summary. To tell the user something while it goes on
    working, the model calls the built-in ``report_result`` tool, whose text
    asks to be spoken. The instruction appended to its prompt says all this.
    ``transform_output`` can change the flag, or the text, or drop the output.

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

        Raises:
            ValueError: If the LLM service declines the role.
        """
        if objection := llm.llm_with_backend_role_objection("backend"):
            raise ValueError(objection)
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
        self._attached: _AttachedFrontend | None = None
        self._transform_output = transform_output
        self.llm.append_system_instruction(BACKEND_OUTPUT_INSTRUCTIONS)
        # A built-in tool: sent on every inference beside the context's own
        # tools and never part of the context's tool set.
        self.llm.register_function(REPORT_TOOL_NAME, self._report_result)
        self.llm.get_llm_adapter().builtin_tools[REPORT_TOOL_NAME] = FunctionSchema(
            name=REPORT_TOOL_NAME,
            description=(
                "Tell the user something now, while you go on working: the result of a "
                "request, or a question. Call it in the same message as the tool calls that "
                "go on with other work. A message with no tool calls is told to the user "
                "as it is, so this is not needed there."
            ),
            properties={"text": {"type": "string", "description": "What to tell the user."}},
            required=["text"],
        )
        # Whether the model's current turn made function calls, which decides
        # whether the text it wrote is a report or notes on the work. The
        # calls are announced before the turn ends, so the flag is read and
        # cleared as the turn stops.
        self._turn_made_calls = False

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
        # The index in the context of a request appended while the model was
        # busy, which asked for no run of its own: the run that follows the
        # current step takes it up, and the model sees the request beside
        # whatever that step produced. Running it separately would run the
        # model twice on the same context, the second time with nothing new
        # to react to. Cleared once a run has it in view.
        self._request_awaiting_run: int | None = None

        @self.llm.event_handler("on_before_process_frame")
        async def on_before_llm_frame(llm, frame: Frame):
            if isinstance(frame, LLMContextFrame):
                self._runs_requested += 1
                if (
                    self._request_awaiting_run is not None
                    and len(frame.context.messages) > self._request_awaiting_run
                ):
                    self._request_awaiting_run = None

        @self.user_aggregator.event_handler("on_before_process_frame")
        async def on_before_user_aggregator_frame(aggregator, frame: Frame):
            if isinstance(frame, LLMMessagesAppendFrame) and self._requests_pending:
                self._requests_pending -= 1
                if not frame.run_llm:
                    self._request_awaiting_run = len(self.context.messages)

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

        @self.assistant_aggregator.event_handler("on_after_process_frame")
        async def on_after_assistant_aggregator_frame(aggregator, frame: Frame):
            # A result that asks for no run may have settled the last call a
            # waiting request was held for; one that asks for a run brings
            # the run itself.
            if (
                isinstance(frame, FunctionCallResultFrame)
                and frame.properties is not None
                and frame.properties.run_llm is False
            ):
                await self._run_awaiting_request()

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
            or self._model_busy
            or self.assistant_aggregator.has_function_calls_in_progress
        )

    @property
    def _model_busy(self) -> bool:
        """Whether a run is in progress or queued for the model."""
        return self._runs_requested > self._runs_completed or self.llm.has_queued_frame(
            LLMContextFrame
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
        await self._run_awaiting_request()
        await self.send_job_response(message.job_id, {"cancelled": was_working})

    async def _refuse(self, job_id: str, reason: str) -> None:
        logger.warning(f"Worker '{self.name}': refusing job {job_id}: {reason}")
        await self.send_job_response(job_id, {"error": reason}, status=JobStatus.ERROR)

    async def _queue_request(self, request: str) -> None:
        """Append a request to the backend's conversation, and run the model on it if it is idle.

        While a run is in progress or queued, or a synchronous call is in
        flight, the request asks for no run of its own: the run that follows
        the current step takes it up (see ``_request_awaiting_run``). A run
        while a synchronous call is in flight would see the call without its
        result, which the adapter cannot send as it is. The frame is
        uninterruptible, so a request queued just ahead of a cancellation
        survives the interruption the cancellation broadcasts.
        """
        self._requests_pending += 1
        frame = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": request}],
            run_llm=not self._model_busy and not self._calls_block_a_run,
        )
        frame.interruptible = False
        await self.queue_frame(frame)

    @property
    def _calls_block_a_run(self) -> bool:
        """Whether a synchronous call is in flight, whose result will bring the next run."""
        return self.assistant_aggregator.has_blocking_function_calls_in_progress

    async def _run_awaiting_request(self) -> None:
        """Run the model on a request that was waiting for the current step, if nothing else will."""
        if (
            self._request_awaiting_run is not None
            and not self._model_busy
            and not self._calls_block_a_run
        ):
            await self.queue_frame(LLMRunFrame())

    async def _stop_work(self, reason: str) -> None:
        """Interrupt the pipeline and cancel every function call in flight.

        An interruption stops the run in progress and the calls that opt into
        cancellation on interruption; the async ones, which an interruption
        leaves running by design, are cancelled on their own.
        """
        await self.queue_frame(InterruptionFrame())
        await self.llm.cancel_function_calls(reason=reason)

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    async def send_output(
        self, output: BackendOutput, *, apply_transform_output: bool = False
    ) -> None:
        """Send one output of the app's own to the frontend.

        It reaches the frontend as the model's outputs do, but as given unless
        asked: ``transform_output`` is not applied to it by default. With no
        frontend attached there is nowhere for it to go, and it is dropped with
        a warning.

        Args:
            output: The output.
            apply_transform_output: Whether to run the output through
                ``transform_output`` as the model's outputs are.
        """
        if self._attached is None:
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
        if self._attached is None:
            return
        try:
            if text:
                await self._emit(BackendOutput(text=text, prefers_spoken=not made_calls))
        except Exception as e:
            logger.error(f"Worker '{self.name}': transform_output failed: {e}")
            await self._send_update({"type": ERROR_UPDATE_TYPE, "error": str(e)})
        await self._run_awaiting_request()
        if not self.working:
            await self._send_update({"type": IDLE_UPDATE_TYPE})

    async def _report_result(self, params: FunctionCallParams):
        """Send what the model wants the user told, as a spoken output."""
        text = str(params.arguments.get("text") or "").strip()
        if text:
            await self._emit(BackendOutput(text=text, prefers_spoken=True))
        # The model goes on with the calls it made beside this one; the report
        # alone is no reason to run it again.
        await params.result_callback(
            {"status": "reported"}, properties=FunctionCallResultProperties(run_llm=False)
        )

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
            # The report tool is the stream's own; its output is already sent.
            if call.function_name != REPORT_TOOL_NAME:
                await self._send_update(call.to_payload())

    async def on_job_cancelled(self, message: BusJobCancelMessage) -> None:
        """Stop the model: the frontend that was attached is gone.

        The ``attach`` job's handler is already cancelled by the time this
        runs, and has let go of the attachment; the work in flight stops too,
        so nothing of it reaches the next frontend.
        """
        request = self.active_jobs.get(message.job_id)
        if request is not None and request.job_name == ATTACH_JOB_NAME:
            await self._stop_work("frontend detached")

    async def _on_pipeline_error(self, frame: ErrorFrame):
        """Tell the frontend the backend cannot go on with what it was doing.

        A tool handler that raises is the exception. The LLM service reports
        that as ``ErrorCategory.APPLICATION``, settles the call with an error
        result and carries on, so the model still has something coming.
        """
        if frame.category == ErrorCategory.APPLICATION:
            return
        if self._attached is not None:
            await self._send_update({"type": ERROR_UPDATE_TYPE, "error": frame.error})

    async def _on_assistant_thought(self, message: AssistantThoughtMessage):
        text = (message.content or "").strip()
        if text and self._attached is not None:
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
        """Send a job update to the attached frontend."""
        if self._attached is not None:
            await self.send_job_update(self._attached.job_id, payload)


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
        #: Whether the stream ended because this side let go of the backend
        #: (the session was closed, or the attaching worker stopped), as
        #: opposed to the backend going away.
        self.detached = False

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
                    response = self._group.responses.get(self._backend_name)
                    if response is None:
                        # Cancelled from this side: the session was closed, or
                        # the requester's worker stopped.
                        self.detached = True
                        raise StopAsyncIteration
                    raise JobError(_with_reason(str(e), response)) from e
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
