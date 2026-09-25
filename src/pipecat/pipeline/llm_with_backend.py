#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An LLM with a backend: a conversational frontend delegating the rest.

The frontend is any LLM service, text or speech-to-speech, and holds the
conversation. The backend is a :class:`~pipecat.workers.llm.backend_llm_worker.BackendLLMWorker`
running a heavier model with the tools, and does the work the frontend hands
off. :class:`LLMWithBackend` wraps the frontend so the pair drops into a
pipeline where an LLM goes, and installs the tools that join them.

The two exchange messages. The frontend's ``delegate`` tool puts a message
to the backend and returns at once; what the backend has to say comes back
as a stream of outputs, each appended to the frontend's conversation as a
message marked ``Backend:``, spoken or silent as the backend's flag says. A
second request joins the first, a correction changes it, and
``cancel_delegated_work`` stops it. How a request
is worded is the :class:`BackendRequestStrategy`'s business, and the
:class:`BackendConnector` owns the tools, the session with the backend, and
how each output is rendered into the conversation.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import FunctionCallResultProperties, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContextMessage
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm.backend_llm_worker import (
    _DEFAULT_TRANSCRIPT_INSTRUCTION,
    _REPORT_INSTRUCTION,
    BackendError,
    BackendEvent,
    BackendIdle,
    BackendLLMWorker,
    BackendOutput,
    BackendToolCall,
    _BackendSession,
    _render_transcript_request,
)

#: Name of the tool the frontend calls to delegate.
DELEGATE_TOOL_NAME = "delegate"

#: Name of the tool the frontend calls to stop the backend's work.
CANCEL_TOOL_NAME = "cancel_delegated_work"

#: How a backend output is marked in the frontend's conversation.
BACKEND_MESSAGE_PREFIX = "Backend: "

#: How a backend reasoning summary is marked in the frontend's conversation.
BACKEND_THOUGHT_PREFIX = "Backend (thinking): "

#: How a function call the backend is making is marked in the frontend's conversation.
BACKEND_WORKING_PREFIX = "Backend (working): "

#: Every mark a rendered backend message can open with.
BACKEND_PREFIXES = (BACKEND_MESSAGE_PREFIX, BACKEND_THOUGHT_PREFIX, BACKEND_WORKING_PREFIX)

#: Carried on each spoken backend message, so the frontend relays it on whatever
#: turn it lands on, a user's turn included, where the standing instruction
#: alone leaves it unsaid.
RELAY_NOTE = (
    "(Relay this to the user now, in your own words, once, even if the conversation has "
    "moved on. If the user has just said something you have not answered yet, answer that "
    "first and relay this at the end of the same reply.)"
)

#: When the frontend delegates and when it does not, ahead of each request
#: strategy's own guidance. The frontend's own system instruction says what
#: the backend is for.
_DELEGATION_POLICY = (
    "Delegate to the backend when any part of what the user asks needs a backend tool or "
    "careful reasoning, or a correction changes work already requested; a request that "
    "arrives while the backend is already working is delegated like any other, since the "
    "backend hears nothing you do not hand over. Do not delegate "
    "when you can answer from the conversation or a result you already have, or when you "
    "need a brief clarification first. Delegate before giving any answer that depends on "
    "backend work, and do not guess the result while waiting. To delegate, call the "
    "delegate tool at once, in that same reply, and do the rest of the reply yourself "
    "around it. "
)

#: What the frontend does with the backend's messages and the stop tool.
_MESSAGES_INSTRUCTION = (
    "After delegating, acknowledge briefly, without offering updates or asking whether to "
    "go ahead, and do whatever else the user asked that you can do yourself. The result says "
    "whether the backend was idle or already working; if it was working, your request joins "
    "that work. The backend sees nothing of the conversation but what you delegate, so every "
    "new request that needs it takes a delegate call of its own, even while it is still "
    "working on an earlier one.\n\n"
    "BACKEND MESSAGES: The backend works on its own after you delegate and may be doing "
    "several things at once. What it has to say arrives as messages in the conversation "
    f'marked "{BACKEND_MESSAGE_PREFIX.strip()}": a result, a question for the user, or news '
    "worth an update, on whatever turn happens to be in progress by then. Such a message is "
    "owed to the user, whatever the conversation has moved on to: small talk, other "
    "questions or a long wait do not make it unwanted. If the user has said something you "
    "have not answered yet, answer that first, then relay the message at the end of that "
    "same reply, in your own words. If you have already answered everything the user said, "
    "relay just the message; do not repeat or rephrase your earlier reply. Relay each "
    "message once. Only when the user has cancelled or changed what they asked for does a "
    "message go unsaid, and then say only what still helps. Messages marked "
    f'"{BACKEND_WORKING_PREFIX.strip()}" and "{BACKEND_THOUGHT_PREFIX.strip()}" are what '
    "the backend is doing and thinking. Never relay them on their own, but when the user "
    "asks how the work is going, answer from the latest of them, concretely: name the step, "
    "such as which file it is reading or that it is running the tests, rather than saying "
    "only that it is still working. Never state a result you have not received from the "
    "backend, and never say work is done that the backend has not said is done.\n\n"
    f"Whenever the user says to stop or cancel what the backend is doing, call "
    f"{CANCEL_TOOL_NAME} at once, in that same reply, even if they ask for something else in "
    "the same breath; then delegate the new request as well. Delegating the stop is not "
    "enough, because the stop must be immediate. Only when the user wants part of the work "
    "kept, or wants it changed rather than stopped, delegate that instead."
)


@dataclass
class ConnectorContext:
    """What a connector needs to know about the layers it joins.

    Parameters:
        backend_name: Name of the backend worker, local or registered elsewhere.
        frontend_is_realtime: Whether the frontend is a speech-to-speech
            service, whose context can lag the audio, which changes how a
            delegation words its request.
    """

    backend_name: str
    frontend_is_realtime: bool


# ---------------------------------------------------------------------------
# Request strategies: frontend → backend
# ---------------------------------------------------------------------------


class BackendRequestStrategy:
    """Defines the ``delegate`` tool the frontend calls, and turns a call into the request the backend receives.

    The tool's parameters are this strategy's to declare, so the connector's
    handler has one signature and reads whatever was declared back from
    ``params.arguments`` in :meth:`compose_request`.
    """

    #: JSON-schema properties of the ``delegate`` tool; empty when it takes none.
    tool_parameters: dict[str, Any] = {}
    #: Which of the properties are required.
    tool_required: list[str] = []
    #: Guidance appended to the frontend's system instruction, if any.
    frontend_instruction: str | None = None

    def tool_description(self) -> str:
        """Describe the ``delegate`` tool to the frontend model.

        Returns:
            The description.
        """
        raise NotImplementedError

    async def compose_request(self, params: FunctionCallParams) -> str | None:
        """Turn a ``delegate`` call into the text put to the backend.

        Args:
            params: The call, carrying the arguments the tool declared and the
                frontend's context.

        Returns:
            The request text, or ``None`` when the call carries nothing new
            for the backend, in which case nothing is sent.
        """
        raise NotImplementedError


class TranscriptBackendRequestStrategy(BackendRequestStrategy):
    """Hands the backend the conversation and lets it work out the request.

    The frontend model calls the tool with no arguments; the request is the
    transcript of what was said since the previous delegation (the whole
    conversation the first time), rendered by
    :func:`~pipecat.workers.llm.backend_llm_worker._render_transcript_request`.
    The backend's own messages in the conversation are left out: the backend
    already has what it said. The default for a text frontend, whose context
    is current when the tool runs.

    The cursor assumes the conversation accrues. A rewritten context or a
    failed delegation's turns are not re-sent; the user's next request
    carries what matters. A call that finds no new turn since the previous
    delegation sends nothing.
    """

    frontend_instruction = _DELEGATION_POLICY + (
        "A delegation hands over the whole conversation, not one item: the backend reads "
        "it and does everything in it that is its job, so delegate once per reply however "
        "many things the user asked for, and do not word the request."
    )

    def __init__(self, *, instruction: str = _DEFAULT_TRANSCRIPT_INSTRUCTION):
        """Initialize the strategy.

        Args:
            instruction: What the backend should do with the transcript, placed
                after it.
        """
        self._instruction = instruction
        self._delegated_through = 0

    def tool_description(self) -> str:
        """Describe the tool: hand the conversation over."""
        return (
            "Hand the conversation over to the backend, which reads it and does everything "
            "in it that needs a backend tool or careful reasoning. Call this as soon as any "
            "part of what the user asks needs that, and do the rest yourself. One handoff "
            "per reply, however many things the user asked for: two questions, or one "
            "question about two places, is one handoff. The backend sees nothing but what "
            "is handed over, so a request the user makes after your last handoff needs a "
            "handoff of its own, even while the backend is still working, and so does a "
            "correction to work already handed over. It takes no arguments. Keep talking "
            "with the user while it works."
        )

    async def compose_request(self, params: FunctionCallParams) -> str | None:
        """Render the turns since the previous delegation as the request."""
        messages = params.context.get_messages()
        if self._delegated_through > len(messages):
            # The context was reset since the previous delegation.
            self._delegated_through = 0
        conversation = [
            m for m in messages[self._delegated_through :] if not _is_backend_message(m)
        ]
        first = self._delegated_through == 0
        self._delegated_through = len(messages)
        if not any(m.get("role") in ("user", "assistant") for m in conversation):  # type: ignore[union-attr]
            return None
        return _render_transcript_request(conversation, instruction=self._instruction, first=first)


class ExplicitBackendRequestStrategy(BackendRequestStrategy):
    """Has the frontend model word the request itself.

    The tool takes a ``request`` argument, which is sent to the backend as it
    stands. The default for a speech-to-speech frontend, which responds to
    audio before the transcript of that audio reaches its context, so the
    turns that prompted a handoff may not be recorded when the tool runs.
    """

    tool_parameters = {
        "request": {
            "type": "string",
            "description": (
                "What the user is asking for now, self-contained: their goal, the details "
                "they gave (places, dates, names) and their latest correction. Work already "
                "handed over stays with the backend; a new request joins it, so do not "
                "restate earlier requests."
            ),
        }
    }
    tool_required = ["request"]
    frontend_instruction = _DELEGATION_POLICY + (
        "Word the request so it stands on its own: the user's goal, the exact details "
        "they gave and their latest correction, with everything new they asked for in the "
        "one request, so you delegate once per reply however many things that is. Work "
        "already handed over stays with the backend: a new request joins it, so do not "
        "restate earlier requests, and a correction names what changes."
    )

    def tool_description(self) -> str:
        """Describe the tool: hand a worded request over."""
        return (
            "Hand a request to the backend, which does everything in it that needs a "
            "backend tool or careful reasoning. Call this as soon as any part of what the "
            "user asks needs that, with the request worded to stand on its own, and do the "
            "rest yourself. One call per reply, however many things the user asked for: two "
            "questions, or one question about two places, go in one request. Work already "
            "handed over stays with the backend, so send only what is new. Keep talking "
            "with the user while it works."
        )

    async def compose_request(self, params: FunctionCallParams) -> str | None:
        """Send the model's request as it stands, with the reporting expectation after it."""
        request = str(params.arguments.get("request") or "").strip()
        return f"{request}\n\n{_REPORT_INSTRUCTION}" if request else None


def _is_backend_message(message: LLMContextMessage) -> bool:
    """Whether a context message is one the connector rendered from a backend output."""
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    return message.get("role") == "developer" and (
        isinstance(content, str) and content.startswith(BACKEND_PREFIXES)
    )


# ---------------------------------------------------------------------------
# The connector
# ---------------------------------------------------------------------------


class BackendConnector:
    """Joins the two layers: the tools the frontend calls, the session with the backend, and what each output becomes.

    A request strategy defines the ``delegate`` tool and composes the request.
    It may be given, or left to the connector to pick by frontend kind once
    bound: :class:`TranscriptBackendRequestStrategy` for a text frontend,
    :class:`ExplicitBackendRequestStrategy` for a speech-to-speech one, whose
    context lags the audio.

    The connector holds the session with the backend for the frontend's life,
    and turns each output the backend sends into a message appended to the
    frontend's conversation with :meth:`render_output`, run or not as the
    output's ``prefers_spoken`` flag says. The function calls the backend makes
    on the way are reported in the frontend's pipeline for clients to show;
    nothing else in the pipeline sees them.

    Example::

        connector = BackendConnector(respond_on_delegate=False)
    """

    def __init__(
        self,
        *,
        request_strategy: BackendRequestStrategy | None = None,
        respond_on_delegate: bool = True,
        timeout_secs: float | None = 30,
    ):
        """Initialize the connector.

        Args:
            request_strategy: How a ``delegate`` call becomes the backend's
                request. Picked by frontend kind when omitted.
            respond_on_delegate: Whether the frontend runs again as soon as a
                delegation is sent, which is where its acknowledgement comes
                from: a text model's tool-call turn carries no prose. Off for
                a frontend that announces a handoff itself.
            timeout_secs: How long the backend may take to acknowledge a
                message or a cancellation, including the wait for it to become
                ready.
        """
        self._request_strategy = request_strategy
        self._respond_on_delegate = respond_on_delegate
        self._timeout_secs = timeout_secs
        self._context: ConnectorContext | None = None
        self._tools: list[FunctionSchema] = []
        self._session: _BackendSession | None = None
        self._session_open = asyncio.Event()

    @property
    def request_strategy(self) -> BackendRequestStrategy:
        """The request strategy in use. Available once bound."""
        assert self._request_strategy is not None, "connector not bound"
        return self._request_strategy

    @property
    def tools(self) -> list[FunctionSchema]:
        """The tools to install on the frontend. Available once bound."""
        assert self._tools, "connector not bound"
        return self._tools

    @property
    def tool(self) -> FunctionSchema:
        """The ``delegate`` tool. Available once bound."""
        return self.tools[0]

    @property
    def frontend_instruction(self) -> str | None:
        """Guidance for the frontend model. Available once bound."""
        parts = [self.request_strategy.frontend_instruction, _MESSAGES_INSTRUCTION]
        return "\n\n".join(p for p in parts if p) or None

    @property
    def session(self) -> _BackendSession | None:
        """The session with the backend, while one is open."""
        return self._session

    def bind(self, context: ConnectorContext) -> None:
        """Settle the strategy for the layers being joined and build the tools.

        Args:
            context: The layers.
        """
        self._context = context
        if self._request_strategy is None:
            self._request_strategy = (
                ExplicitBackendRequestStrategy()
                if context.frontend_is_realtime
                else TranscriptBackendRequestStrategy()
            )
        self._tools = self.build_tools()

    def build_tools(self) -> list[FunctionSchema]:
        """Build the tools to install on the frontend: ``delegate`` first, then ``cancel_delegated_work``.

        Override to install tools of another shape entirely; the service reads
        nothing else from the connector but :attr:`frontend_instruction`.

        Returns:
            The tools, each carrying its handler.
        """

        async def delegate(params: FunctionCallParams):
            await self.delegate(params)

        async def cancel(params: FunctionCallParams):
            await self.cancel(params)

        return [
            FunctionSchema(
                name=DELEGATE_TOOL_NAME,
                description=self.request_strategy.tool_description(),
                properties=self.request_strategy.tool_parameters,
                required=self.request_strategy.tool_required,
                handler=delegate,
            ),
            FunctionSchema(
                name=CANCEL_TOOL_NAME,
                description=(
                    "Stop all the work the backend is doing now. Call this whenever the user "
                    "says to stop, cancel, or never mind, even if they ask for something else "
                    "in the same breath: call this, and delegate the new request too. Only "
                    "when the user wants part of the work kept, or wants it changed rather "
                    "than stopped, delegate that instead. Returns at once."
                ),
                properties={},
                required=[],
                handler=cancel,
            ),
        ]

    async def run_session(self, worker: BaseWorker, frontend: LLMService[Any]) -> None:
        """Hold the session with the backend, delivering what it produces, until cancelled.

        Args:
            worker: The pipeline worker the frontend runs in.
            frontend: The frontend service, which takes the deliveries.
        """
        assert self._context is not None, "connector not bound"
        try:
            async with _BackendSession(
                worker, self._context.backend_name, timeout_secs=self._timeout_secs
            ) as session:
                self._session = session
                self._session_open.set()
                async for event in session:
                    await self.deliver(frontend, event)
                if not session.detached:
                    logger.warning(f"Backend '{self._context.backend_name}' went away")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"The session with backend '{self._context.backend_name}' failed: {e}")
        finally:
            self._session = None
            self._session_open.clear()

    async def _open_session(self) -> _BackendSession:
        """The session, once it is open; waits for it up to the timeout."""
        if self._session is None:
            try:
                await asyncio.wait_for(self._session_open.wait(), self._timeout_secs)
            except TimeoutError:
                raise RuntimeError("the backend is not attached") from None
        assert self._session is not None
        return self._session

    async def delegate(self, params: FunctionCallParams) -> None:
        """Put the request a ``delegate`` call carries to the backend, and settle the call at once.

        Args:
            params: The ``delegate`` call.
        """
        assert self._context is not None, "connector not bound"
        request = await self.request_strategy.compose_request(params)
        if request is None:
            # Nothing new since the previous delegation: the backend has it all.
            logger.debug(f"Delegate call {params.tool_call_id} carries nothing new; not sent")
            await params.result_callback(
                {"status": "already_delegated"},
                properties=FunctionCallResultProperties(run_llm=False),
            )
            return
        logger.debug(f"Delegating to '{self._context.backend_name}': {request!r}")
        session = await self._open_session()
        status = await session.send(request)
        await params.result_callback(
            {"status": "delegated", "backend": status},
            properties=FunctionCallResultProperties(run_llm=self._respond_on_delegate),
        )

    async def cancel(self, params: FunctionCallParams) -> None:
        """Stop the backend's work, and settle the call with whether there was any.

        Args:
            params: The ``cancel_delegated_work`` call.
        """
        session = await self._open_session()
        cancelled = await session.cancel("cancelled by the user")
        await params.result_callback(
            {"status": "cancelled" if cancelled else "nothing_running"},
            properties=FunctionCallResultProperties(run_llm=True),
        )

    async def deliver(self, frontend: LLMService[Any], event: BackendEvent) -> None:
        """Deliver one thing the backend produced to the frontend.

        An output becomes a message appended to the frontend's conversation,
        which runs the frontend when the output asks to be spoken. A function
        call phase is reported as the
        :class:`~pipecat.frames.frames.ExternalFunctionCallFrame` for it, which
        the RTVI observer turns into function-call events. An error becomes a
        spoken message, so the user hears the work stopped. The appended
        messages are uninterruptible: one queued while the bot speaks waits
        behind that speech, and an interruption clears the speech but must
        leave the message to be recorded.

        Args:
            frontend: The frontend service.
            event: What the backend produced.
        """
        if isinstance(event, BackendOutput):
            logger.debug(
                f"Backend output ({'spoken' if event.prefers_spoken else 'silent'}"
                f"{', thought' if event.is_thought else ''}): {event.text!r}"
            )
            message = self.render_output(event)
            if message is not None:
                await self._append(
                    frontend,
                    LLMMessagesAppendFrame(messages=[message], run_llm=event.prefers_spoken),
                )
        elif isinstance(event, BackendToolCall):
            await frontend.push_frame(event.to_frame())
            message = self.render_tool_call(event)
            if message is not None:
                await self._append(
                    frontend, LLMMessagesAppendFrame(messages=[message], run_llm=False)
                )
        elif isinstance(event, BackendError):
            logger.warning(f"Backend error: {event.error}")
            await self._append(
                frontend,
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "developer",
                            "content": (
                                f"{BACKEND_MESSAGE_PREFIX}The work could not be completed."
                                f"\n\n{RELAY_NOTE}"
                            ),
                        }
                    ],
                    run_llm=True,
                ),
            )
        elif isinstance(event, BackendIdle):
            logger.debug("Backend is idle")

    async def _append(self, frontend: LLMService[Any], frame: LLMMessagesAppendFrame) -> None:
        """Queue a message into the frontend's conversation, past any interruption on the way."""
        frame.interruptible = False
        await frontend.queue_frame(frame)

    def render_output(self, output: BackendOutput) -> LLMContextMessage | None:
        """Render a backend output as the message the frontend's conversation takes in.

        Override for another wording, role, or to leave some outputs out by
        returning ``None``. The default marks the message so the frontend can
        tell it from the user's and its own, and so the transcript request
        strategy can leave it out of what it sends the backend; a spoken
        output carries :data:`RELAY_NOTE` as well.

        Args:
            output: The output.

        Returns:
            The message, or ``None`` to deliver nothing for this output.
        """
        if not output.text:
            return None
        if output.is_thought:
            return {"role": "developer", "content": f"{BACKEND_THOUGHT_PREFIX}{output.text}"}
        note = f"\n\n{RELAY_NOTE}" if output.prefers_spoken else ""
        return {"role": "developer", "content": f"{BACKEND_MESSAGE_PREFIX}{output.text}{note}"}

    def render_tool_call(self, call: BackendToolCall) -> LLMContextMessage | None:
        """Render a phase of a backend function call as a message the frontend's conversation takes in, silently.

        The default records each call as it starts, with its arguments, so the
        frontend can say what the backend is doing if asked; the other phases
        are not recorded. Override to record more, less, or nothing.

        Args:
            call: The phase of the call.

        Returns:
            The message, or ``None`` to record nothing for this phase.
        """
        if call.phase != "in_progress":
            return None
        arguments = ", ".join(f"{k}={v!r}" for k, v in (call.arguments or {}).items())
        return {
            "role": "developer",
            "content": f"{BACKEND_WORKING_PREFIX}{call.function_name}({arguments})",
        }


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class LLMWithBackend(Pipeline):
    """A conversational LLM, kept fast and light, with a backend that does the work it hands off.

    **Alpha.** The API is unstable: names, arguments, the strategies and the
    job contract are likely to change between releases.

    Put it where the LLM goes in a pipeline. It is a :class:`Pipeline`, not an
    :class:`LLMService`: functions, settings updates, event handlers and a
    ``FlowManager`` go on :attr:`frontend`. It wraps the frontend service,
    installs the connector's tools on it as built-in tools, appends the
    connector's guidance to the frontend's system instruction, adds a local
    backend worker to the pipeline worker so the app never wires it up, and
    holds the session with the backend for the pipeline's life. A built-in
    tool is sent on every inference beside whatever tools the frontend has,
    and never enters the context's tool set, so a tool change announced to
    the model never mentions it. The tools are the backend's; the frontend
    has ``delegate`` and ``cancel_delegated_work`` and no more, though a tool
    it must keep, in its context or configured on the service, stays.

    The guidance says when to delegate in general terms. The frontend's own
    system instruction is the place to say what the backend is for, in plain
    words, as the backend's own instruction does: no tool names, nothing to
    update when a tool changes.

    Example::

        llm = LLMWithBackend(
            frontend=OpenAILLMService(...),
            backend=BackendLLMWorker(
                llm=AnthropicLLMService(...),
                context=LLMContext(tools=[get_weather]),
            ),
        )
        pipeline = Pipeline([transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg])
    """

    def __init__(
        self,
        *,
        frontend: LLMService[Any],
        backend: BackendLLMWorker | str,
        connector: BackendConnector | None = None,
    ):
        """Initialize the service.

        Args:
            frontend: The conversational LLM service, text or speech-to-speech.
            backend: The backend worker, or the name of one registered
                elsewhere (in the app, or in another process on a shared bus).
                A worker given here is added to the pipeline worker at setup.
            connector: How delegation works. A default
                :class:`BackendConnector` when omitted.

        Raises:
            ValueError: If the frontend service declines the role.
        """
        logger.warning("LLMWithBackend is alpha: its API is likely to change between releases.")
        if objection := frontend.llm_with_backend_role_objection("frontend"):
            raise ValueError(objection)
        self._frontend = frontend
        self._backend = backend
        self._connector = connector or BackendConnector()
        self._connector.bind(
            ConnectorContext(
                backend_name=backend if isinstance(backend, str) else backend.name,
                frontend_is_realtime=frontend.service_metadata_frame().is_realtime_service,
            )
        )
        if instruction := self._connector.frontend_instruction:
            frontend.append_system_instruction(instruction)
        for tool in self._connector.tools:
            frontend.register_function(tool.name, tool.handler, cancel_on_interruption=False)
            frontend.get_llm_adapter().builtin_tools[tool.name] = tool
        self._session_task: asyncio.Task | None = None
        super().__init__([frontend])

    @property
    def frontend(self) -> LLMService[Any]:
        """The frontend LLM service."""
        return self._frontend

    @property
    def backend(self) -> BackendLLMWorker | str:
        """The backend worker, or its name."""
        return self._backend

    @property
    def connector(self) -> BackendConnector:
        """The connector joining the two layers."""
        return self._connector

    @property
    def tool(self) -> FunctionSchema:
        """The ``delegate`` tool installed on the frontend."""
        return self._connector.tool

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the frontend, register a local backend worker as a child of the pipeline worker, and open the session."""
        await super().setup(setup)
        if isinstance(self._backend, BackendLLMWorker):
            await self.pipeline_worker.add_workers(self._backend)
        self._session_task = self.create_task(
            self._connector.run_session(self.pipeline_worker, self._frontend),
            f"{self}::backend_session",
        )

    async def cleanup(self):
        """Close the session with the backend, then clean up the frontend."""
        if self._session_task is not None:
            await self.cancel_task(self._session_task)
            self._session_task = None
        await super().cleanup()
