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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from loguru import logger

from pipecat.bus.messages import BusJobCancelMessage, BusJobRequestMessage
from pipecat.frames.frames import (
    ErrorFrame,
    ExternalFunctionCall,
    ExternalFunctionCallCancelFrame,
    ExternalFunctionCallInProgressFrame,
    ExternalFunctionCallResultFrame,
    ExternalFunctionCallsStartedFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
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
from pipecat.services.llm_service import LLMService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm.llm_context_worker import LLMContextWorker

#: Name of the job that attaches a frontend to a :class:`BackendLLMWorker`.
ATTACH_JOB_NAME = "attach"

#: Name of the job that puts a message to an attached :class:`BackendLLMWorker`.
MESSAGE_JOB_NAME = "message"

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

#: The mark the backend's model puts at the start of a message it wants the
#: user told. The message is sent with ``prefers_spoken=True``, mark stripped;
#: a message without it is a note (``prefers_spoken=False``). Strict prompt,
#: flexible parsing: the model is told to put the mark first, but sometimes
#: writes a note paragraph and then a marked one, so a mark at the start of a
#: later line splits the message, a note before it and a spoken message from
#: it on, rather than losing the result. The turn-completion markers are
#: read the same way (``user_turn_completion_mixin``).
SPOKEN_MARK = ">>"

#: Where a message's spoken part starts: the first line that begins with the mark.
_SPOKEN_MARK_LINE = re.compile(rf"(?m)^[ \t]*{re.escape(SPOKEN_MARK)}[ \t]*")

#: How the backend's model works with the assistant, appended to its system
#: instruction: who it is, which of what it writes the user hears, and how it
#: takes requests. The backend's own system instruction says what it does;
#: this says how what it writes reaches the user.
#: ``BackendLLMWorker(pairing_instruction=...)`` replaces it.
BACKEND_PAIRING_INSTRUCTION = (
    "You are the backend of a voice assistant. The assistant talks with the user and sends "
    "you requests: the conversation it is having, or a request it worded for you. Do the "
    "parts that need your tools or careful reasoning; the assistant handles the rest of the "
    "conversation itself, such as small talk, jokes and stories, so leave those to it.\n\n"
    f"WHAT THE USER HEARS: Begin a message with {SPOKEN_MARK} and the whole of it is told to "
    "the user, in the assistant's own words, tool calls or not. Anything you write without "
    "the mark is not told to the user: the assistant keeps it as notes on your work, and "
    "draws on them only if the user asks how the work is going. "
    "Mark a message to give a result, to ask a question you cannot proceed without, or for "
    "news the user should have now. A result you do not mark is "
    f"lost: the message that gives it begins with {SPOKEN_MARK}. Never say again what you "
    "have already marked, and never mark that you have stopped or that there is nothing to "
    "do: the assistant has already told the user. Write a marked message as plain spoken "
    "sentences the assistant can say as they are: no headings, lists, Markdown or raw "
    "JSON.\n\n"
    "REQUESTS: A request may arrive while you are working on an earlier one. It may add "
    "work, change it, or stop some or all of it: follow the latest, cancel the tools whose "
    "results are no longer wanted, and do not repeat work already done. With several "
    "requests in hand, do the newest first unless it depends on an earlier one, and give "
    "each one's result as soon as you have it, not when everything is done."
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
    clients, as the ``ExternalFunctionCall*Frame`` :meth:`to_frame` builds, and
    does nothing else with it. The phases mirror
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

    def to_frame(self) -> Frame:
        """Build the frame that reports this phase in the frontend's pipeline.

        Returns:
            The frame for the phase.
        """
        if self.phase == "started":
            # The frame can announce several calls at once, as the pipeline's
            # own FunctionCallsStartedFrame does; a backend reports its calls
            # one at a time, so here it carries one.
            return ExternalFunctionCallsStartedFrame(
                [ExternalFunctionCall(self.function_name, self.tool_call_id)]
            )
        if self.phase == "in_progress":
            return ExternalFunctionCallInProgressFrame(
                self.function_name, self.tool_call_id, arguments=self.arguments
            )
        if self.phase == "result":
            return ExternalFunctionCallResultFrame(
                self.function_name,
                self.tool_call_id,
                arguments=self.arguments,
                result=self.result,
                is_final=self.is_final,
            )
        return ExternalFunctionCallCancelFrame(self.function_name, self.tool_call_id)


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
TRANSCRIPT_REQUEST_INSTRUCTION = (
    "Act on the user's most recent request in the conversation above. If it asks for "
    "something, tell the user the result as soon as you have it, before going on with other "
    f"work, in a message that begins with {SPOKEN_MARK}. If it only stops or changes work "
    "already under way, tell them nothing: the assistant already has."
)

#: Appended to a request the frontend worded itself, so the backend tells the
#: user its result on its own rather than with whatever else it is doing.
EXPLICIT_REQUEST_INSTRUCTION = (
    "If this request asks for something, tell the user the result as soon as you have it, "
    f"before going on with other work, in a message that begins with {SPOKEN_MARK}. If it "
    "only stops or changes work already under way, tell them nothing: the assistant already "
    "has."
)


def _render_explicit_request(
    request: str, *, instruction: str = EXPLICIT_REQUEST_INSTRUCTION
) -> str:
    """Render a request the frontend worded itself, with what the backend should do with it after.

    Args:
        request: The request, as the frontend's model worded it.
        instruction: What the backend should do with the request, placed
            after it.

    Returns:
        The rendered request.
    """
    return f"{request}\n\n{instruction}"


def _split_spoken(text: str) -> tuple[str, str]:
    """Split a message the model wrote into its note and its spoken part, by the mark.

    Args:
        text: The message, stripped.

    Returns:
        The note (what precedes the first line that begins with the mark, or
        the whole message when no line does) and the spoken part (what follows
        the mark, with any later mark removed, or empty), each stripped.
    """
    match = _SPOKEN_MARK_LINE.search(text)
    if match is None:
        return text, ""
    # From the first marked line on, the whole of it is spoken; a mark the
    # model put on a later line is not meant for the user's ear.
    spoken = _SPOKEN_MARK_LINE.sub("", text[match.end() :])
    return text[: match.start()].strip(), spoken.strip()


def _render_transcript_request(
    conversation: Sequence[LLMContextMessage],
    *,
    instruction: str = TRANSCRIPT_REQUEST_INSTRUCTION,
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

    The model decides what the user hears: a message it begins with
    :data:`SPOKEN_MARK` asks to be spoken, whether or not the turn also calls
    tools, and the mark is stripped; anything else it writes, reasoning
    summaries included, is a note that the frontend keeps but does not speak.
    A mark at the start of a later line splits the message into a note and
    a spoken part, since the model sometimes writes it that way.
    The backend's own system instruction says what it does and, in plain
    words, what the user should be told; the pairing instruction the worker
    appends (:data:`BACKEND_PAIRING_INSTRUCTION`, or the ``pairing_instruction``
    given) is the only place the mark is named. ``transform_output`` can
    change the flag, or the text, or drop the output.

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
        pairing_instruction: str = BACKEND_PAIRING_INSTRUCTION,
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
            pairing_instruction: How the model works with the assistant,
                appended to the LLM's system instruction:
                :data:`BACKEND_PAIRING_INSTRUCTION` unless given. One of its
                own has to say what that one says: that the assistant sends
                it requests, that the user hears what it marks with
                :data:`SPOKEN_MARK` and nothing else, and how to take requests
                that arrive while it works.
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
        self.llm.append_system_instruction(pairing_instruction)

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
        # Whether the idle update has been sent for the current lull, so a
        # call that settles after the last turn stopped does not send it twice.
        self._idle_announced = False
        # The index in the context of a request appended while the model was
        # busy, as an LLMMessagesAppendFrame with run_llm=False: the run that
        # follows the current step takes it up, and the model sees the request
        # alongside whatever that step produced. Running it separately would
        # run the model twice on the same context, the second time with
        # nothing new to react to. Cleared once an LLMContextFrame that holds
        # the request reaches the LLM.
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
            elif isinstance(frame, InterruptionFrame):
                # The run in progress, if any, is cancelled by this frame and
                # ends as an interrupted turn, which is not counted as
                # completed; a run queued behind it is dropped. A run that
                # starts after the frame has passed is a new one, so the
                # counters are reset here, as the LLM sees the interruption,
                # and not later at the assistant aggregator, where a reset
                # could erase a run that had just started.
                self._runs_requested = self._runs_completed = 0

        @self.user_aggregator.event_handler("on_before_process_frame")
        async def on_before_user_aggregator_frame(aggregator, frame: Frame):
            if isinstance(frame, LLMMessagesAppendFrame) and self._requests_pending:
                self._requests_pending -= 1
                if not frame.run_llm:
                    self._request_awaiting_run = len(self.context.messages)
                    # The synchronous call the request was held for may have
                    # settled by the time the request lands in the context, as
                    # when the model cancelled it with its cancel_<tool> tool
                    # meanwhile, so check again whether the request can run now.
                    await self._run_awaiting_request()

        # The backend's function calls are relayed as they pass the assistant
        # aggregator, which every phase of a call reaches.
        @self.assistant_aggregator.event_handler("on_before_process_frame")
        async def on_before_assistant_aggregator_frame(aggregator, frame: Frame):
            await self._on_function_call_frame(frame)

        @self.assistant_aggregator.event_handler("on_after_process_frame")
        async def on_after_assistant_aggregator_frame(aggregator, frame: Frame):
            # A FunctionCallResultFrame with run_llm=False may have settled the
            # last synchronous call a waiting request was held for, so the
            # request runs now; one with run_llm=True brings the run itself.
            if (
                isinstance(frame, FunctionCallResultFrame)
                and frame.properties is not None
                and frame.properties.run_llm is False
            ) or isinstance(frame, FunctionCallCancelFrame):
                await self._run_awaiting_request()
                # A call that settles (its result or cancellation arrives)
                # after the last turn stopped may have been the last thing the
                # backend had in flight, so the idle update is sent from here
                # as well as from the end of a turn.
                await self._announce_idle_if_done()

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
        return {"steering": True, "progress": True}

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
            # The attach job ends only when the frontend cancels it, which is
            # how the frontend detaches.
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

    async def _refuse(self, job_id: str, reason: str) -> None:
        logger.warning(f"Worker '{self.name}': refusing job {job_id}: {reason}")
        await self.send_job_response(job_id, {"error": reason}, status=JobStatus.ERROR)

    async def _queue_request(self, request: str) -> None:
        """Append a request to the backend's conversation, and run the model on it if it is idle.

        While a run is in progress or queued, or a synchronous call is in
        flight, the request asks for no run of its own: the run that follows
        the current step takes it up (see ``_request_awaiting_run``). A run
        while a synchronous call is in flight would see the call without its
        result, which the adapter cannot send as it is.
        """
        self._requests_pending += 1
        self._idle_announced = False
        run_llm = not self._model_busy and not self._synchronous_call_in_flight
        if run_llm:
            logger.debug(f"Worker '{self.name}': running the model on the request")
        else:
            logger.debug(
                f"Worker '{self.name}': holding the request for the current step "
                f"(model busy={self._model_busy}, synchronous call in flight={self._synchronous_call_in_flight})"
            )
        frame = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": request}],
            run_llm=run_llm,
        )
        await self.queue_frame(frame)

    @property
    def _synchronous_call_in_flight(self) -> bool:
        """Whether a synchronous call is in flight, whose result will bring the next run."""
        return self.assistant_aggregator.has_synchronous_function_calls_in_progress

    async def _run_awaiting_request(self) -> None:
        """Run the model on a request that was waiting for the current step, if nothing else will."""
        if (
            self._request_awaiting_run is not None
            and not self._model_busy
            and not self._synchronous_call_in_flight
        ):
            logger.debug(f"Worker '{self.name}': running the model on the request that was held")
            await self.queue_frame(LLMRunFrame())

    async def _stop_work(self, reason: str) -> None:
        """Interrupt the pipeline and cancel every function call in flight.

        An interruption stops the run in progress and the calls that opt into
        cancellation on interruption; the async ones, which an interruption
        leaves running by design, are cancelled on their own. A request not
        yet taken up is dropped with the interruption, and one held for the
        current step is let go of, so that neither runs the model on work the
        frontend has abandoned: the cancellations that follow would otherwise
        run a held request as they settle.
        """
        self._requests_pending = 0
        self._request_awaiting_run = None
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
        # Not passed through transform_output by default
        # (apply_transform_output=False), so an app can silence the model's
        # outputs with a transform_output that drops them and still send
        # outputs of its own, such as from a tool the model calls to talk to
        # the user.
        await self._emit(output, apply_transform_output=apply_transform_output)

    async def _on_assistant_turn_stopped(self, message: AssistantTurnStoppedMessage):
        if message.interrupted:
            # Text from an interrupted turn is not sent, and the run counters
            # were reset as the LLM saw the InterruptionFrame.
            return
        self._runs_completed += 1
        text = (message.content or "").strip()
        if self._attached is None:
            return
        try:
            note, spoken = _split_spoken(text)
            if note:
                await self._emit(BackendOutput(text=note, prefers_spoken=False))
            if spoken:
                await self._emit(BackendOutput(text=spoken, prefers_spoken=True))
        except Exception as e:
            logger.error(f"Worker '{self.name}': transform_output failed: {e}")
            await self._send_update({"type": ERROR_UPDATE_TYPE, "error": str(e)})
        await self._run_awaiting_request()
        await self._announce_idle_if_done()

    async def _announce_idle_if_done(self) -> None:
        """Send the idle update once the backend has nothing left to do, once per lull."""
        if self._attached is None or self.working or self._idle_announced:
            return
        self._idle_announced = True
        await self._send_update({"type": IDLE_UPDATE_TYPE})

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
        """Stop the model: the frontend that was attached is gone.

        The ``attach`` job's handler is already cancelled by the time this
        runs, and has let go of the attachment; the work in flight stops too,
        so none of it is reported after the frontend has let go of it.
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
    iterator over what the backend produces, and :meth:`send` puts messages
    to it. The iteration ends when the backend
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
            timeout_secs: How long a message may take to be acknowledged,
                including the wait for the backend to become ready. The
                attachment itself has no timeout.
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
