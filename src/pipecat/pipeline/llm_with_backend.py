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
second request joins the first, and a correction changes or stops it. How a
request is worded is the :class:`BackendRequestStrategy`'s business, and the
:class:`BackendConnector` owns the tools, the session with the backend, and
how each output is rendered into the conversation.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.classifiers.base_classifier import BaseClassifier, YesNoQuestion
from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    FunctionCallResultProperties,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    MetricsFrame,
    SystemFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContextMessage
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.llm_service import FunctionCallParams, LLMService, _OutputGate
from pipecat.services.settings import LLMSettings
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm.backend_llm_worker import (
    EXPLICIT_REQUEST_INSTRUCTION,
    TRANSCRIPT_REQUEST_INSTRUCTION,
    BackendError,
    BackendEvent,
    BackendIdle,
    BackendLLMWorker,
    BackendOutput,
    BackendToolCall,
    _BackendSession,
    _render_explicit_request,
    _render_transcript_request,
)

#: Name of the tool the frontend calls to delegate.
DELEGATE_TOOL_NAME = "delegate"


#: How a backend output is marked in the frontend's conversation.
BACKEND_MESSAGE_PREFIX = "Backend: "

#: How a backend reasoning summary is marked in the frontend's conversation.
BACKEND_THOUGHT_PREFIX = "Backend (thinking): "

#: How a backend note, text its model wrote without the mark that asks for
#: the user to be told it, is marked in the frontend's conversation.
BACKEND_NOTE_PREFIX = "Backend (note): "

#: How a function call the backend is making is marked in the frontend's conversation.
BACKEND_WORKING_PREFIX = "Backend (working): "

#: Every mark a rendered backend message can open with.
BACKEND_PREFIXES = (
    BACKEND_MESSAGE_PREFIX,
    BACKEND_THOUGHT_PREFIX,
    BACKEND_NOTE_PREFIX,
    BACKEND_WORKING_PREFIX,
)

# ---------------------------------------------------------------------------
# Prompts
#
# What follows is read by the frontend's model, and the pairing instruction
# and the tool descriptions are also read by predictive delegation's
# classifier. A small change in wording can change what a model does in ways
# no unit test catches: gpt-4.1 stops delegating a "never mind", or the
# classifier starts delegating questions about how the work is going. Before
# and after changing any of these, run the checks in
# scripts/llm-with-backend/tuning/README.md, and record their results in
# RESULTS.md beside it.
# ---------------------------------------------------------------------------

#: Carried on each spoken backend message, so the frontend relays it on whatever
#: turn it lands on, a user's turn included, where the standing instruction
#: alone leaves it unsaid.
RELAY_NOTE = (
    "(Relay this to the user now, in your own words, once, even if the conversation has "
    "moved on. If the user has just said something you have not answered yet, answer that "
    "first and relay this at the end of the same reply.)"
)

#: Where the request strategy's own guidance goes in :data:`FRONTEND_PAIRING_INSTRUCTION`.
STRATEGY_PLACEHOLDER = "{strategy}"

#: How the frontend's model works with a backend, appended to its system
#: instruction: what the backend is and when to delegate, that to the user
#: there is one assistant, what to do after delegating, and what to do with
#: the backend's messages. The request strategy's guidance (how the request
#: is worded) goes where :data:`STRATEGY_PLACEHOLDER` stands, at the end of
#: the first paragraph. The frontend's own system instruction says what the
#: backend is for. ``BackendConnector(frontend_pairing_instruction=...)``
#: replaces it.
#:
#: The instruction ends by restating its one rule, that a message about
#: delegated work, a stop included, gets a delegate call. A stop is where a
#: model goes wrong: it answers "never mind" with "I won't make any changes"
#: and no call, and the backend carries on. gpt-4.1 weighs the end of the
#: instruction most, so the restatement is what makes it delegate a stop;
#: the first paragraph alone, with the rule in it, does not, nor does the
#: rule as a list of cases.
#: Before changing it, see "Prompts" above.
FRONTEND_PAIRING_INSTRUCTION = (
    "THE BACKEND: You have a backend for anything that needs a backend tool or careful "
    "reasoning. It sees nothing of the conversation but what you delegate, and nothing you "
    "say reaches it: to start, change or stop its work, call the delegate tool, at once, in "
    "the same reply, and do the rest of the reply yourself around it. That holds for every "
    "message that concerns its work, whether a new request, a yes to something that was "
    'offered, a correction or a "never mind", and even while earlier work is still running: '
    "a stop you only speak leaves the work running. Do not delegate what you can do "
    "yourself: small talk, a question that a result you already have answers, or a question "
    "about how the work is going, which you answer from the backend's latest notes or, with "
    "none yet, by saying there is no update yet. Never guess a result while waiting. "
    f"{STRATEGY_PLACEHOLDER}\n\n"
    "ONE ASSISTANT: To the user there is one assistant, you: never mention the backend, "
    "delegation, a handoff or tools. Say what is being done as your own doing, such as \"I'm "
    'checking", never "the backend is checking", and never that you passed anything along.\n\n'
    "AFTER DELEGATING: Acknowledge in one short sentence of your own that fits the request, "
    "naming the task briefly if that reads naturally. For example: "
    '"Let me look into that", "Sure, I\'ll get on that now" or "Okay, pulling that up now". An '
    "acknowledgement is never a substitute for the delegate call. In the acknowledgement, avoid "
    "restating the request in full, or describing in detail what you are going to do, or "
    "following up on your own with promises, offers or questions: you've fired off the delegate "
    "call, so let it be. "
    "Then answer whatever else the user asked that you can. Never say work is done, or give a "
    "result, that the backend has not given you; if the user asks whether work is under way and "
    "you never delegated it, say so and delegate it then.\n\n"
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
    f'"{BACKEND_WORKING_PREFIX.strip()}", "{BACKEND_NOTE_PREFIX.strip()}" and '
    f'"{BACKEND_THOUGHT_PREFIX.strip()}" are what the backend is doing and thinking. They '
    "carry no results: what a call found reaches you only in a message marked "
    f'"{BACKEND_MESSAGE_PREFIX.strip()}", so never describe or guess what a call found. Never '
    "relay them on their own, but when the user asks how the work is going, answer from the "
    "latest of them, concretely: name the step, such as which file it is reading or that it "
    "is running the tests, rather than saying only that it is still working. Never state a "
    "result you have not received from the backend, and never say work is done that the "
    "backend has not said is done.\n\n"
    "Whatever else a reply does, a user message about work you delegated, whether to add to "
    "it, change it or stop it, a never mind included, gets a delegate call in that reply: "
    "nothing else stops or changes that work."
)

#: The transcript strategy's guidance, in the pairing instruction's delegating paragraph.
#: Before changing it, see "Prompts" above.
TRANSCRIPT_FRONTEND_INSTRUCTION = (
    "A delegation hands over the whole conversation, not one item: the backend reads it and "
    "does everything in it that is its job, so delegate once per reply however many things "
    "the user asked for, and do not word the request."
)

#: How the transcript strategy describes the ``delegate`` tool.
#: Before changing it, see "Prompts" above.
TRANSCRIPT_TOOL_DESCRIPTION = (
    "The only way to start, change or stop the backend's work. Hands the conversation over to "
    "the backend, which reads it and does everything in it that needs a backend tool or "
    "careful reasoning, and stops or changes what the user has called off or changed. Call it "
    "as soon as any part of what the user says concerns that work, even while the backend is "
    "still working, and do the rest yourself. Not for a question about how the work is going, "
    "which you answer from the backend's notes. One handoff per reply, however many things the "
    "user asked for. It takes no arguments. Keep talking with the user while it works."
)

#: The explicit strategy's guidance, in the pairing instruction's delegating paragraph.
#: Before changing it, see "Prompts" above.
EXPLICIT_FRONTEND_INSTRUCTION = (
    "Word the request so it stands on its own: the user's goal, the exact "
    "details they gave and their latest correction, with everything new they asked for in "
    "the one request, so you delegate once per reply however many things that is. Work "
    "already handed over stays with the backend: a new request joins it, so do not restate "
    "earlier requests; a correction names what changes, and a stop names what to stop."
)

#: How the explicit strategy describes the ``delegate`` tool.
#: Before changing it, see "Prompts" above.
EXPLICIT_TOOL_DESCRIPTION = (
    "The only way to start, change or stop the backend's work. Hands a request to the "
    "backend, which does everything in it that needs a backend tool or careful reasoning, "
    "and stops or changes what the user has called off or changed. Call it as soon as any "
    "part of what the user says concerns that work, even while the backend is still working, "
    "with the request worded to stand on its own, and do the rest yourself. Not for a question "
    "about how the work is going, which you answer from the backend's notes. One call per "
    "reply, however many things the user asked for. Work already handed over stays with the "
    "backend, so send only what is new, including anything the user says about that work, "
    "such as to stop or change it. Keep talking with the user while it works."
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
#
# Two ways for a delegate call to become the backend's request, chosen by
# frontend kind when the app gives none. TranscriptBackendRequestStrategy
# hands over the conversation since the previous delegation: the default
# for a text frontend, whose context is current when the tool runs.
# ExplicitBackendRequestStrategy has the frontend model word the request
# itself: the default for a speech-to-speech frontend, which responds to
# audio before the transcript of that audio reaches its context, so the
# turns that prompted a handoff may not be in the context yet, and a
# transcript taken then would miss them.
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
    #: The strategy's guidance to the frontend model, if any: how the request
    #: is worded. It goes where the pairing instruction's
    #: :data:`STRATEGY_PLACEHOLDER` stands.
    frontend_instruction: str | None = None
    #: How the ``delegate`` tool is described to the frontend model.
    tool_description: str = ""
    #: Whether the strategy reads the conversation from the frontend's context,
    #: so it needs that context to be current when the ``delegate`` tool runs.
    #: A speech-to-speech frontend's context lags the audio, so the connector
    #: refuses such a strategy there.
    requires_context: bool = False
    #: Whether a ``delegate`` call has to carry arguments for the request to
    #: be built. Predictive delegation makes its ``delegate`` call with none,
    #: so it works only with a strategy that sets this False.
    delegate_needs_arguments: bool = True

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

    requires_context = True
    delegate_needs_arguments = False

    def __init__(
        self,
        *,
        frontend_instruction: str | None = TRANSCRIPT_FRONTEND_INSTRUCTION,
        tool_description: str = TRANSCRIPT_TOOL_DESCRIPTION,
        request_instruction: str = TRANSCRIPT_REQUEST_INSTRUCTION,
    ):
        """Initialize the strategy.

        Args:
            frontend_instruction: The strategy's guidance to the frontend
                model, placed in the pairing instruction: that a delegation
                hands over the whole conversation, unworded.
            tool_description: How the ``delegate`` tool is described.
            request_instruction: What the backend should do with the
                transcript, placed after it.
        """
        self.frontend_instruction = frontend_instruction
        self.tool_description = tool_description
        self._request_instruction = request_instruction
        self._delegated_through = 0

    async def compose_request(self, params: FunctionCallParams) -> str | None:
        """Render the turns since the previous delegation as the request."""
        messages = params.context.get_messages()
        if self._delegated_through > len(messages):
            # The context holds fewer messages than the previous delegation
            # read: it was reset since, so the transcript starts from the top.
            self._delegated_through = 0
        # An LLMSpecificMessage, in a service's own format, is not a user or
        # assistant turn, so the transcript leaves it out.
        conversation = [
            m
            for m in messages[self._delegated_through :]
            if isinstance(m, dict) and not _is_backend_message(m)
        ]
        first = self._delegated_through == 0
        self._delegated_through = len(messages)
        if not any(m.get("role") in ("user", "assistant") for m in conversation):
            return None
        return _render_transcript_request(
            conversation, instruction=self._request_instruction, first=first
        )


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
                "they gave (places, dates, names) and their latest correction, or what they "
                "said about work already under way, such as to stop it or change it. Work "
                "already handed over stays with the backend; a new request joins it, so do "
                "not restate earlier requests."
            ),
        }
    }
    tool_required = ["request"]

    def __init__(
        self,
        *,
        frontend_instruction: str | None = EXPLICIT_FRONTEND_INSTRUCTION,
        tool_description: str = EXPLICIT_TOOL_DESCRIPTION,
        request_instruction: str = EXPLICIT_REQUEST_INSTRUCTION,
    ):
        """Initialize the strategy.

        Args:
            frontend_instruction: The strategy's guidance to the frontend
                model, placed in the pairing instruction: how to word the
                request.
            tool_description: How the ``delegate`` tool is described.
            request_instruction: What the backend should do with the request,
                placed after it.
        """
        self.frontend_instruction = frontend_instruction
        self.tool_description = tool_description
        self._request_instruction = request_instruction

    async def compose_request(self, params: FunctionCallParams) -> str | None:
        """Send the model's request as it stands, with the reporting expectation after it."""
        request = str(params.arguments.get("request") or "").strip()
        if not request:
            return None
        return _render_explicit_request(request, instruction=self._request_instruction)


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
    context lags the audio. A given strategy that reads the frontend's
    context (:attr:`BackendRequestStrategy.requires_context`) is refused for
    a speech-to-speech frontend.

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
        frontend_pairing_instruction: str = FRONTEND_PAIRING_INSTRUCTION,
        respond_on_delegate: bool = True,
        timeout_secs: float | None = 30,
        client_trace: bool = False,
    ):
        """Initialize the connector.

        Args:
            request_strategy: How a ``delegate`` call becomes the backend's
                request. Picked by frontend kind when omitted.
            frontend_pairing_instruction: How the model works with a backend,
                appended to its system instruction:
                :data:`FRONTEND_PAIRING_INSTRUCTION` unless given. One of its
                own has to say what that one says: when and how to delegate,
                that to the user there is one assistant, what to do after
                delegating, and what to do with the backend's messages. The
                request strategy's guidance goes where
                :data:`STRATEGY_PLACEHOLDER` stands, or after the text when
                it has none.
            respond_on_delegate: Whether the frontend runs again as soon as a
                delegation is sent, which is where its acknowledgement comes
                from: a text model's tool-call turn carries no prose. Off for
                a frontend that announces a handoff itself.
            timeout_secs: How long the backend may take to acknowledge a
                message, including the wait for it to become ready.
            client_trace: Whether to send every exchange with the backend to
                the client as an RTVI server message (``type``
                ``"llm-with-backend"``): each request, output, tool-call phase,
                error and idle. For debugging; a client's event
                log shows them.
        """
        self._request_strategy = request_strategy
        self._frontend_pairing_instruction = frontend_pairing_instruction
        self._respond_on_delegate = respond_on_delegate
        self._timeout_secs = timeout_secs
        self._client_trace = client_trace
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
        """What is appended to the frontend's system instruction: the pairing instruction with the strategy's guidance in place. Available once bound."""
        pairing = self._frontend_pairing_instruction
        guidance = self.request_strategy.frontend_instruction or ""
        if STRATEGY_PLACEHOLDER in pairing:
            return pairing.replace(STRATEGY_PLACEHOLDER, guidance).replace(" \n", "\n").strip()
        return "\n\n".join(p for p in (pairing, guidance) if p) or None

    @property
    def session(self) -> _BackendSession | None:
        """The session with the backend, while one is open."""
        return self._session

    def bind(self, context: ConnectorContext) -> None:
        """Settle the strategy for the layers being joined and build the tools.

        Args:
            context: The layers.

        Raises:
            ValueError: If the frontend is speech-to-speech and the strategy
                reads the frontend's context, which lags the audio there.
        """
        self._context = context
        if self._request_strategy is None:
            self._request_strategy = (
                ExplicitBackendRequestStrategy()
                if context.frontend_is_realtime
                else TranscriptBackendRequestStrategy()
            )
        if context.frontend_is_realtime and self._request_strategy.requires_context:
            raise ValueError(
                f"{type(self._request_strategy).__name__} reads the conversation from the "
                "frontend's context, which lags the audio on a speech-to-speech frontend. "
                "Use a request strategy that doesn't read the context, such as "
                "ExplicitBackendRequestStrategy."
            )
        self._tools = self.build_tools()

    def build_tools(self) -> list[FunctionSchema]:
        """Build the tools to install on the frontend: ``delegate``.

        Override to install tools of another shape entirely; the service reads
        nothing else from the connector but :attr:`frontend_instruction`.

        Returns:
            The tools, each carrying its handler.
        """

        async def delegate(params: FunctionCallParams):
            await self.delegate(params)

        return [
            FunctionSchema(
                name=DELEGATE_TOOL_NAME,
                description=self.request_strategy.tool_description,
                properties=self.request_strategy.tool_parameters,
                required=self.request_strategy.tool_required,
                handler=delegate,
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
            # No turns since the previous delegation: the backend already has
            # everything the transcript would carry, so nothing is sent.
            logger.debug(f"Delegate call {params.tool_call_id} carries nothing new; not sent")
            await params.result_callback(
                {"status": "already_delegated"},
                properties=FunctionCallResultProperties(run_llm=False),
            )
            return
        logger.debug(f"Delegating to '{self._context.backend_name}': {request!r}")
        session = await self._open_session()
        status = await session.send(request)
        await self._trace(params.llm, event="request", text=request, backend=status)
        await params.result_callback(
            {"status": "delegated", "backend": status},
            properties=FunctionCallResultProperties(run_llm=self._respond_on_delegate),
        )

    async def deliver(self, frontend: LLMService[Any], event: BackendEvent) -> None:
        """Deliver one thing the backend produced to the frontend.

        An output becomes a message appended to the frontend's conversation,
        which runs the frontend when the output asks to be spoken. A function
        call phase is reported as the
        ``ExternalFunctionCall*Frame`` for it, which
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
            await self._trace(
                frontend,
                event="output",
                text=event.text,
                spoken=event.prefers_spoken,
                thought=event.is_thought,
            )
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
            await self._trace(
                frontend,
                event="tool_call",
                phase=event.phase,
                name=event.function_name,
                arguments=event.arguments,
                result=event.result,
            )
            await frontend.push_frame(event.to_frame())
            message = self.render_tool_call(event)
            if message is not None:
                await self._append(
                    frontend, LLMMessagesAppendFrame(messages=[message], run_llm=False)
                )
        elif isinstance(event, BackendError):
            await self._trace(frontend, event="error", error=event.error)
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
            await self._trace(frontend, event="idle")
            logger.debug("Backend is idle")

    async def _trace(self, frontend: LLMService[Any], **data: Any) -> None:
        """Send one exchange with the backend to the client, when tracing is on."""
        if self._client_trace:
            await frontend.push_frame(
                RTVIServerMessageFrame(data={"type": "llm-with-backend", **data})
            )

    async def _append(self, frontend: LLMService[Any], frame: LLMMessagesAppendFrame) -> None:
        """Queue a message into the frontend's conversation, past any interruption on the way."""
        frame.interruptible = False
        await frontend.queue_frame(frame)

    def render_output(self, output: BackendOutput) -> LLMContextMessage | None:
        """Render a backend output as the message the frontend's conversation takes in.

        Override for another wording, role, or to leave some outputs out by
        returning ``None``. The default marks the message so the frontend can
        tell it from the user's and its own, and so the transcript request
        strategy can leave it out of what it sends the backend: a spoken
        output as ``Backend:`` with :data:`RELAY_NOTE` after it, a silent one
        as ``Backend (note):``, a reasoning summary as ``Backend (thinking):``.

        Args:
            output: The output.

        Returns:
            The message, or ``None`` to deliver nothing for this output.
        """
        if not output.text:
            return None
        if output.is_thought:
            return {"role": "developer", "content": f"{BACKEND_THOUGHT_PREFIX}{output.text}"}
        if not output.prefers_spoken:
            return {"role": "developer", "content": f"{BACKEND_NOTE_PREFIX}{output.text}"}
        return {
            "role": "developer",
            "content": f"{BACKEND_MESSAGE_PREFIX}{output.text}\n\n{RELAY_NOTE}",
        }

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
# Predictive delegation
#
# Predictive delegation (PredictiveDelegation, passed as
# `LLMWithBackend(predictive_delegation=...)`) is an optimization that can
# speed up responses that need backend involvement. Normally, a text frontend
# spends a whole inference deciding to call `delegate` before the backend can
# start. Predictive delegation uses a fast yes/no classifier to determine
# whether it needs to wait for that inference or can go ahead and trigger an
# early backend delegation.
#
# How it works: when a user's turn ends, the frontend runs in parallel (i.e.
# in a race) with the classifier (see `pipecat.classifiers`) that decides
# whether the turn needs the backend.
#
# - If the classifier says "yes" before the frontend has produced output:
#   - the frontend's inference is cancelled;
#   - a `delegate` call to the backend is made right away, in its place.
# - Otherwise, the frontend handles the turn as usual, having waited for
#   nothing (note that it may still choose to `delegate`).
#
# _PredictiveDelegationRace tracks whether a classifier "yes" arrives before the
# frontend's first output, and so whether the turn is delegated early or
# handled by the frontend as usual. A classifier "no" plays no part. The
# frontend is normally idle when a turn ends (the user's speech interrupts
# it), so its inference starts at once; a "yes" that comes while it is still
# busy is held until it starts. The race is attached to the frontend as its
# output gate (`_OutputGate` in `pipecat.services.llm_service`), which the
# frontend consults from its process_frame, push_frame and run_function_calls.
# ---------------------------------------------------------------------------


class _PredictiveDelegationRaceState(Enum):
    """Where the predictive delegation race for a user's turn stands.

    - ``PENDING``: the frontend has been asked to reply to the user's turn,
      but hasn't started its inference yet: normally for only a moment, longer
      if it's still busy with an earlier response. A classifier "yes" now is
      held until the inference starts.
    - ``RACING``: the frontend's inference is running and hasn't produced any
      output yet. A classifier "yes" now, or one held from ``PENDING``,
      delegates the turn early.
    - ``FRONTEND_WON``: the frontend's inference produced output before any
      classifier "yes", or the turn was interrupted. The frontend handles the
      turn as usual.
    - ``CLASSIFIER_WON``: a classifier "yes" came first. The turn is delegated
      early, and the frontend's inference is cancelled.
    """

    PENDING = "pending"
    RACING = "racing"
    FRONTEND_WON = "frontend won"
    CLASSIFIER_WON = "classifier won"


@dataclass
class _EarlyDelegationFrame(SystemFrame):
    """Asks the frontend to delegate a user's turn now, in place of its own inference.

    Sent by PredictiveDelegation when its classifier says "yes". If the
    frontend's inference for the turn hasn't produced anything yet, the
    frontend makes the given calls and cancels the inference; otherwise the
    frame does nothing. A system frame, so the frontend handles it at once,
    while the inference runs on another task.

    Parameters:
        turn: The context frame of the user's turn.
        function_calls: The calls made in place of the inference (the
            ``delegate`` call).
    """

    turn: LLMContextFrame | None = None
    function_calls: list[FunctionCallFromLLM] = field(default_factory=list)


class _PredictiveDelegationRace(_OutputGate):
    """Tracks whether a classifier "yes" arrives before the frontend's first output for a turn.

    See "Predictive delegation" above. Early delegation makes the
    frame's calls, then cancels the frontend's inference with
    ``FrameProcessor._cancel_frame_in_progress``, which keeps the frontend's
    queue, including the acknowledgement run the call starts. Downstream sees
    an ordinary response holding the call; any other output the inference
    pushes before it's cancelled is dropped.

    The frontend's inference is recognized by the task its frames and calls
    come from, the frontend's process task; everything else the frontend
    pushes is left alone. All checks are synchronous, so the frontend's first
    output and early delegation can't interleave.
    """

    def __init__(self):
        """Initialize with no race started."""
        self._turn: LLMContextFrame | None = None
        # The frontend's process task, which runs its inference for the turn
        # and goes on to process the frames after it.
        self._frontend_inference_task: asyncio.Task | None = None
        # None until the first user turn starts a race; after that, the latest
        # race's state, which stays put once decided until the next turn.
        self._state: _PredictiveDelegationRaceState | None = None
        self._frontend: LLMService[Any] | None = None
        # A "yes" that arrived while PENDING, held until the inference starts
        # its response.
        self._held_early_delegation: _EarlyDelegationFrame | None = None

    def start(self, frame: LLMContextFrame) -> None:
        """Start a race for a user's turn, just before the turn is passed to the frontend.

        Args:
            frame: The context frame of the user's turn.
        """
        self._turn = frame
        self._frontend_inference_task = None
        self._state = _PredictiveDelegationRaceState.PENDING
        self._held_early_delegation = None

    async def process_frame(self, service: LLMService[Any], frame: Frame) -> None:
        """Follow the race through the frames the frontend processes.

        Implements ``_OutputGate.process_frame``. Racing starts when the
        frontend's inference for the turn starts, an interruption leaves the
        turn to the frontend, and an early delegation frame is carried out
        while racing.

        Args:
            service: The frontend service.
            frame: The frame it is processing.
        """
        if isinstance(frame, LLMContextFrame):
            if frame is self._turn and self._state == _PredictiveDelegationRaceState.PENDING:
                self._frontend = service
                self._frontend_inference_task = asyncio.current_task()
                self._state = _PredictiveDelegationRaceState.RACING
        elif isinstance(frame, InterruptionFrame):
            # The interruption cancels the inference: there is nothing left
            # for early delegation to replace.
            if self._state in (
                _PredictiveDelegationRaceState.PENDING,
                _PredictiveDelegationRaceState.RACING,
            ):
                self._state = _PredictiveDelegationRaceState.FRONTEND_WON
        elif isinstance(frame, _EarlyDelegationFrame):
            await self._delegate_early(service, frame)

    def allow_push(self, frame: Frame) -> bool:
        """Whether the frontend may push a frame; notes its inference's first output.

        Implements ``_OutputGate.allow_push``.

        Args:
            frame: The frame the frontend is about to push.

        Returns:
            Whether the frontend may push the frame: False only for what the
            inference pushes after the turn was delegated early and before its
            cancellation lands, other than the end of its response.
        """
        if not self._called_from_frontend_inference():
            return True
        self._note_frontend_inference_push(frame)
        if self._state == _PredictiveDelegationRaceState.CLASSIFIER_WON:
            return isinstance(frame, (SystemFrame, LLMFullResponseEndFrame))
        return True

    def allow_function_calls(self) -> bool:
        """Whether the frontend may run function calls; notes its inference's first call.

        Implements ``_OutputGate.allow_function_calls``.

        Returns:
            Whether the frontend may run them: False only for calls the
            inference reaches after the turn was delegated early and before
            its cancellation lands.
        """
        if not self._called_from_frontend_inference():
            return True
        self._note_frontend_inference_function_calls()
        return self._state != _PredictiveDelegationRaceState.CLASSIFIER_WON

    def _called_from_frontend_inference(self) -> bool:
        """Whether the caller is the frontend's process task, running its inference for the turn."""
        return (
            self._frontend_inference_task is not None
            and asyncio.current_task() is self._frontend_inference_task
        )

    def _note_frontend_inference_push(self, frame: Frame) -> None:
        """Note a frame the frontend's inference pushes while racing.

        The start of its response carries out an early delegation held from
        ``PENDING``. Any other output but a system frame is the inference's
        first, and leaves the turn to the frontend.
        """
        if self._state != _PredictiveDelegationRaceState.RACING:
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            if self._held_early_delegation and self._frontend:
                # Carried out on the frontend's input task, like any early
                # delegation frame, once this start frame is downstream.
                held, self._held_early_delegation = self._held_early_delegation, None
                self._frontend.create_task(self._frontend.queue_frame(held))
        elif not isinstance(frame, SystemFrame):
            self._state = _PredictiveDelegationRaceState.FRONTEND_WON

    def _note_frontend_inference_function_calls(self) -> None:
        """Note the frontend's inference reaching function calls.

        While racing, its first call leaves the turn to the frontend.
        """
        if self._state == _PredictiveDelegationRaceState.RACING:
            self._state = _PredictiveDelegationRaceState.FRONTEND_WON

    async def _delegate_early(
        self, frontend: LLMService[Any], frame: _EarlyDelegationFrame
    ) -> None:
        """If the turn is racing, make the frame's calls and cancel the inference.

        A frame that arrives while the turn is pending is held until the
        inference starts.
        """
        turn = frame.turn
        if turn is None or turn is not self._turn:
            return
        if self._state == _PredictiveDelegationRaceState.PENDING:
            logger.debug("Predictive delegation: turn pending; holding the early delegation")
            self._held_early_delegation = frame
            return
        if self._state != _PredictiveDelegationRaceState.RACING:
            logger.debug("Predictive delegation: not delegating early (the race is decided)")
            return
        logger.debug("Predictive delegation: delegating early, cancelling the frontend's inference")
        self._state = _PredictiveDelegationRaceState.CLASSIFIER_WON
        await frontend.run_function_calls(frame.function_calls)
        await frontend._cancel_frame_in_progress(turn)


#: How likely a classifier "yes" must be for a turn to be delegated early.
_DELEGATION_PREDICTION_THRESHOLD = 0.6

#: How many of the conversation's latest messages the classifier sees.
_DELEGATION_PREDICTION_HISTORY_LENGTH = 12


class PredictiveDelegation(FrameProcessor):
    """An optimization that can speed up responses that need backend involvement.

    **Alpha.** It uses a fast yes/no classifier to determine, after each user
    request, whether it needs to wait for the frontend's inference or can go
    ahead and trigger an early backend delegation. See "Predictive delegation"
    above for how it works. Pass one to
    ``LLMWithBackend(predictive_delegation=...)``, which puts it just in front
    of the frontend; it needs a text frontend and a request strategy whose
    ``delegate`` call needs no arguments, such as the default
    :class:`TranscriptBackendRequestStrategy`.

    It races every turn that ends with a new user message, spoken or typed,
    and so also a user-role message the app appends and runs the frontend on.
    The classifier is asked whether that message needs the backend, given
    the frontend's system instruction (where the app says what the backend
    is for, and the pairing instruction when to delegate), the ``delegate``
    tool's description, and the latest messages of the conversation, the
    backend's included. A turn the frontend sees as speculative (eager end of
    turn), or with incomplete-turn filtering on, is left to the frontend.

    Example::

        llm = LLMWithBackend(
            frontend=OpenAILLMService(...),
            backend=BackendLLMWorker(...),
            predictive_delegation=PredictiveDelegation(JevClassifier(api_key=...)),
        )
    """

    def __init__(self, classifier: BaseClassifier, **kwargs):
        """Initialize predictive delegation.

        Args:
            classifier: The yes/no classifier to ask. It answers within a bound
                of its own; an answer that comes after the frontend's first
                output is ignored.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._classifier = classifier
        self._race = _PredictiveDelegationRace()
        self._frontend: LLMService[Any] | None = None
        self._question: YesNoQuestion | None = None
        self._last_raced: LLMContextMessage | None = None
        self._classify_task: asyncio.Task | None = None
        self._warned: set[str] = set()

    def _bind(self, frontend: LLMService[Any], connector: BackendConnector) -> None:
        """Attach the race to the frontend and build the classifier's question."""
        self._frontend = frontend
        frontend._output_gate = self._race
        self._question = YesNoQuestion(
            instructions=(
                "Does the user's latest message need to be handed to the assistant's backend "
                "now? The handoff works like this: " + connector.request_strategy.tool_description
            ),
            no="The assistant's instructions say not to hand this message to the backend.",
        )

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the classifier, reporting its metrics in the pipeline."""
        await super().setup(setup)
        await self._classifier.setup(self.task_manager)
        self._classifier.add_event_handler("on_metrics", self._on_classifier_metrics)

    async def cleanup(self):
        """Stop any classification in progress and clean up the classifier."""
        await self._cancel_classification()
        await self._classifier.cleanup()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Race each new user turn; pass every frame on.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            await self._cancel_classification()
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and self._should_start_race(frame)
        ):
            await self._cancel_classification()
            self._race.start(frame)
            state = self._classifier_state(frame)
            await self.push_frame(frame, direction)
            self._classify_task = self.create_task(
                self._classify(frame, state), f"{self}::classify"
            )
            return
        await self.push_frame(frame, direction)

    def _should_start_race(self, frame: LLMContextFrame) -> bool:
        """Whether a context frame should start a race.

        It should when it asks for a reply to a user message that hasn't been
        raced yet, unless the turn is speculative (eager end of turn) or the
        frontend filters incomplete user turns.
        """
        assert self._frontend is not None, "predictive delegation not bound"
        if frame.speculation:
            self._warn_once(
                "speculation",
                "Predictive delegation: speculative turns (eager end of turn) are left to the "
                "frontend",
            )
            return False
        if self._frontend_settings().filter_incomplete_user_turns:
            self._warn_once(
                "filtering",
                "Predictive delegation: turns are left to the frontend while it filters "
                "incomplete user turns",
            )
            return False
        messages = frame.context.get_messages()
        last = messages[-1] if messages else None
        if not isinstance(last, dict) or last.get("role") != "user":
            return False
        if last is self._last_raced:
            return False
        self._last_raced = last
        return True

    def _classifier_state(self, frame: LLMContextFrame) -> dict[str, Any]:
        """What the classifier is told about the turn."""
        # The instruction the frontend's model reads, the pairing instruction
        # included, so the classifier follows the same rules for delegating.
        return {
            "assistant": str(self._frontend_settings().system_instruction or ""),
            "conversation": _render_for_classifier(frame.context.get_messages()),
        }

    async def _classify(self, frame: LLMContextFrame, state: dict[str, Any]) -> None:
        """Ask the classifier about a turn; on a "yes", ask the frontend to delegate it early."""
        assert self._question is not None
        try:
            answers = await self._classifier.yes_no(state, {"needs_backend": self._question})
        except Exception as e:
            logger.warning(f"Predictive delegation: the classifier failed: {e}")
            return
        probability = answers["needs_backend"].probability
        logger.debug(f"Predictive delegation: classifier says {probability:.2f}")
        if probability < _DELEGATION_PREDICTION_THRESHOLD:
            return
        call = FunctionCallFromLLM(
            function_name=DELEGATE_TOOL_NAME,
            tool_call_id=f"early_delegation_{uuid.uuid4().hex[:16]}",
            arguments={},
            context=frame.context,
        )
        await self.push_frame(_EarlyDelegationFrame(turn=frame, function_calls=[call]))

    def _frontend_settings(self) -> LLMSettings:
        assert self._frontend is not None, "predictive delegation not bound"
        settings = self._frontend.settings
        assert isinstance(settings, LLMSettings)
        return settings

    async def _cancel_classification(self) -> None:
        if self._classify_task is not None:
            await self.cancel_task(self._classify_task)
            self._classify_task = None

    async def _on_classifier_metrics(self, classifier: BaseClassifier, data: list[MetricsData]):
        await self.push_frame(MetricsFrame(data=data))

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)


def _render_for_classifier(messages: list[LLMContextMessage]) -> list[dict[str, str]]:
    """The conversation's latest messages, as speaker and text, for the classifier."""
    rendered: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        if role == "user" and isinstance(content, str):
            rendered.append({"speaker": "user", "text": content})
        elif role == "assistant":
            if isinstance(content, str) and content:
                rendered.append({"speaker": "assistant", "text": content})
            for call in message.get("tool_calls") or []:
                if call.get("function", {}).get("name") == DELEGATE_TOOL_NAME:
                    rendered.append(
                        {"speaker": "assistant", "action": "handed the request to the backend"}
                    )
        elif _is_backend_message(message) and isinstance(content, str):
            rendered.append(
                {"speaker": "backend", "text": content.removesuffix(f"\n\n{RELAY_NOTE}")}
            )
    return rendered[-_DELEGATION_PREDICTION_HISTORY_LENGTH:]


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
    has ``delegate`` and no more, though a tool
    it must keep, in its context or configured on the service, stays.

    The guidance is the connector's pairing instruction (how to work with a
    backend: when and how to delegate, and what to do with the backend's
    messages) with the strategy's (how the request is worded) in place. The
    frontend's own system instruction is the place to say what the backend is
    for, in plain words, as the backend's own instruction does: no tool names,
    nothing to update when a tool changes.

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
        predictive_delegation: PredictiveDelegation | None = None,
    ):
        """Initialize the service.

        Args:
            frontend: The conversational LLM service, text or speech-to-speech.
            backend: The backend worker, or the name of one registered
                elsewhere (in the app, or in another process on a shared bus).
                A worker given here is added to the pipeline worker at setup.
            connector: How delegation works. A default
                :class:`BackendConnector` when omitted.
            predictive_delegation: A :class:`PredictiveDelegation`, to speed
                up responses that need backend involvement: its classifier
                determines whether to wait for the frontend's inference or go
                ahead and trigger an early backend delegation. Only for a text
                frontend.

        Raises:
            ValueError: If the frontend service declines the role; if the
                connector's request strategy reads the frontend's context and
                the frontend is speech-to-speech; or if predictive delegation
                is given with a speech-to-speech frontend or with a request
                strategy whose ``delegate`` call needs arguments.
        """
        logger.warning("LLMWithBackend is alpha: its API is likely to change between releases.")
        if objection := frontend.llm_with_backend_role_objection("frontend"):
            raise ValueError(objection)
        self._frontend = frontend
        self._backend = backend
        self._connector = connector or BackendConnector()
        frontend_is_realtime = frontend.service_metadata_frame().is_realtime_service
        self._connector.bind(
            ConnectorContext(
                backend_name=backend if isinstance(backend, str) else backend.name,
                frontend_is_realtime=frontend_is_realtime,
            )
        )
        if predictive_delegation is not None:
            if frontend_is_realtime:
                raise ValueError(
                    "Predictive delegation needs a text frontend: a speech-to-speech frontend "
                    "replies to the user's audio before the turn's transcript reaches its "
                    "context, so there is no turn to classify in time."
                )
            if self._connector.request_strategy.delegate_needs_arguments:
                raise ValueError(
                    f"Predictive delegation makes its delegate call with no arguments, which "
                    f"{type(self._connector.request_strategy).__name__} can't build a request "
                    "from. Use a strategy that needs none, such as "
                    "TranscriptBackendRequestStrategy."
                )
        if instruction := self._connector.frontend_instruction:
            frontend.append_system_instruction(instruction)
        for tool in self._connector.tools:
            frontend.register_function(tool.name, tool.handler, cancel_on_interruption=False)
            frontend.get_llm_adapter().builtin_tools[tool.name] = tool
        self._predictive_delegation = predictive_delegation
        self._session_task: asyncio.Task | None = None
        if predictive_delegation is not None:
            predictive_delegation._bind(frontend, self._connector)
            super().__init__([predictive_delegation, frontend])
        else:
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
    def predictive_delegation(self) -> PredictiveDelegation | None:
        """The PredictiveDelegation in front of the frontend, if there is one."""
        return self._predictive_delegation

    @property
    def tool(self) -> FunctionSchema:
        """The ``delegate`` tool installed on the frontend."""
        return self._connector.tool

    async def setup(self, setup: FrameProcessorSetup):
        """Register a local backend worker as a child of the pipeline worker, set up the frontend, and open the session."""
        # Before setting the frontend up: it gets the StartFrame as soon as its
        # own setup is done, and can delegate from then on.
        if isinstance(self._backend, BackendLLMWorker):
            await setup.pipeline_worker.add_workers(self._backend)
        await super().setup(setup)
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
