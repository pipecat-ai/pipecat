#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The connector, which joins a frontend to its backend.

:class:`BackendConnector` owns the tools the frontend calls, the session with
the backend, and how each backend output is rendered into the frontend's
conversation.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import FunctionCallResultProperties, LLMMessagesAppendFrame
from pipecat.pipeline.llm_with_backend.prompts import (
    BACKEND_MESSAGE_PREFIX,
    BACKEND_MESSAGE_RELAY_INSTRUCTION,
    BACKEND_NOTE_PREFIX,
    BACKEND_THOUGHT_PREFIX,
    BACKEND_WORKING_PREFIX,
    FRONTEND_PAIRING_INSTRUCTION,
    PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER,
)
from pipecat.pipeline.llm_with_backend.request_strategies import (
    BackendRequestStrategy,
    ExplicitBackendRequestStrategy,
    TranscriptBackendRequestStrategy,
)
from pipecat.processors.aggregators.llm_context import LLMContextMessage
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm.backend_llm_worker import (
    BackendError,
    BackendEvent,
    BackendIdle,
    BackendOutput,
    BackendToolCall,
    _BackendSession,
)

#: Name of the tool the frontend calls to delegate.
DELEGATE_TOOL_NAME = "delegate"


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
                :data:`FRONTEND_PAIRING_INSTRUCTION` unless given (both
                constants named here are in
                :mod:`~pipecat.pipeline.llm_with_backend.prompts`). One of its
                own has to say what that one says: when and how to delegate,
                that to the user there is one assistant, what to do after
                delegating, and what to do with the backend's messages. The
                request strategy's guidance goes where
                :data:`PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER` stands, or
                after the text when it has none.
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
        if PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER in pairing:
            return (
                pairing.replace(PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER, guidance)
                .replace(" \n", "\n")
                .strip()
            )
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
                                f"\n\n{BACKEND_MESSAGE_RELAY_INSTRUCTION}"
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
        output as ``Backend:`` with :data:`BACKEND_MESSAGE_RELAY_INSTRUCTION`
        after it, a silent one as ``Backend (note):``, a reasoning summary as
        ``Backend (thinking):``.

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
            "content": (
                f"{BACKEND_MESSAGE_PREFIX}{output.text}\n\n{BACKEND_MESSAGE_RELAY_INSTRUCTION}"
            ),
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
