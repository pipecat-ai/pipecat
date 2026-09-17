#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A dual LLM: a conversational frontend delegating to a backend.

The frontend is any LLM service, text or speech-to-speech, and holds the
conversation. The backend is a :class:`~pipecat.workers.llm.backend_llm_worker.BackendLLMWorker`
running a heavier model with the tools, and does the work the frontend hands
off. :class:`PipecatDualLLMService` wraps the frontend so the pair drops into a
pipeline where an LLM goes, and installs the ``delegate`` tool that joins
them.

How a delegation crosses is the :class:`BackendConnector`'s business, built
from two strategies: a :class:`BackendRequestStrategy` defines the tool the
frontend calls and turns a call into the request the backend receives, and a
:class:`BackendReplyStrategy` turns each thing the backend produces into what
the frontend hears about it. The connector picks defaults by frontend kind:
a text frontend hands over the conversation and relays the backend's
progress as it comes; a speech-to-speech frontend words the request itself
and takes every output at once, since its function calls accept one
result.
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    FunctionCallResultProperties,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import (
    FrameProcessorSetup,
)
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.workers.llm.backend_llm_worker import (
    _DEFAULT_TRANSCRIPT_INSTRUCTION,
    BackendLLMWorker,
    BackendOutput,
    BackendToolCall,
    _BackendFinalOutput,
    _delegate_to_backend,
    _render_transcript_request,
)

#: Name of the tool the frontend calls to delegate.
DELEGATE_TOOL_NAME = "delegate"

#: When the frontend delegates and when it does not, ahead of each request
#: strategy's own guidance. The frontend's own system instruction says what
#: the backend is for.
_DELEGATION_POLICY = (
    "Delegate to the backend when any part of what the user asks needs a backend tool or "
    "careful reasoning, or a correction changes work already requested. Do not delegate "
    "when you can answer from the conversation or a result you already have, or when you "
    "need a brief clarification first. Delegate before giving any answer that depends on "
    "backend work, and do not guess the result while waiting. To delegate, call the "
    "delegate tool at once, in that same reply, and do the rest of the reply yourself "
    "around it. "
)


@dataclass
class ConnectorContext:
    """What a connector needs to know about the layers it joins.

    Parameters:
        backend_name: Name of the backend worker, local or registered elsewhere.
        frontend_is_realtime: Whether the frontend is a speech-to-speech
            service. Its function calls accept one result, and its context can
            lag the audio, which changes what a delegation can send and receive.
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

    async def compose_request(self, params: FunctionCallParams) -> str:
        """Turn a ``delegate`` call into the text put to the backend.

        Args:
            params: The call, carrying the arguments the tool declared and the
                frontend's context.

        Returns:
            The request text.
        """
        raise NotImplementedError


class TranscriptBackendRequestStrategy(BackendRequestStrategy):
    """Hands the backend the conversation and lets it work out the request.

    The frontend model calls the tool with no arguments; the request is the
    transcript of what was said since the previous delegation (the whole
    conversation the first time), rendered by
    :func:`~pipecat.workers.llm.backend_llm_worker._render_transcript_request`.
    The default for a text frontend, whose context is current when the tool
    runs.
    """

    frontend_instruction = _DELEGATION_POLICY + (
        "A delegation hands over the whole conversation, not one item: the backend reads "
        "it and does everything in it that is its job, so delegate once per reply however "
        "many things the user asked for, and do not word the request. While it works, "
        "keep the conversation going. When its result comes back, relay it in your own "
        "words."
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
            "question about two places, is one handoff. It takes no arguments. Keep "
            "talking with the user while it works."
        )

    async def compose_request(self, params: FunctionCallParams) -> str:
        """Render the turns since the previous delegation as the request."""
        messages = params.context.get_messages()
        if self._delegated_through > len(messages):
            # The context was reset since the previous delegation.
            self._delegated_through = 0
        conversation = messages[self._delegated_through :]
        first = self._delegated_through == 0
        self._delegated_through = len(messages)
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
                "The request, self-contained: the user's goal, the details they gave "
                "(places, dates, names) and their latest correction."
            ),
        }
    }
    tool_required = ["request"]
    frontend_instruction = _DELEGATION_POLICY + (
        "Word the request so it stands on its own: the user's goal, the exact details "
        "they gave and their latest correction, with everything they asked for in the one "
        "request, so you delegate once per reply however many things that is. While it "
        "works, keep the conversation going. When its result comes back, relay it in your "
        "own words."
    )

    def tool_description(self) -> str:
        """Describe the tool: hand a worded request over."""
        return (
            "Hand a request to the backend, which does everything in it that needs a "
            "backend tool or careful reasoning. Call this as soon as any part of what the "
            "user asks needs that, with the request worded to stand on its own, and do the "
            "rest yourself. One call per reply, however many things the user asked for: two "
            "questions, or one question about two places, go in one request. Keep talking "
            "with the user while it works."
        )

    async def compose_request(self, params: FunctionCallParams) -> str:
        """Send the model's request as it stands."""
        return str(params.arguments.get("request") or "")


# ---------------------------------------------------------------------------
# Reply strategies: backend → frontend
# ---------------------------------------------------------------------------


class BackendReplyStrategy:
    """Turns each :class:`BackendOutput` into what the frontend hears about it.

    The final output settles the ``delegate`` call; what the frontend hears
    of the outputs before it, and when, is what strategies differ on.
    """

    #: Whether the strategy reports outputs before the final one as
    #: intermediate tool results. A speech-to-speech frontend cannot take
    #: those: its function calls accept one result.
    needs_intermediate_results: bool = False
    #: Guidance appended to the frontend's system instruction, if any.
    frontend_instruction: str | None = None

    async def deliver(
        self, params: FunctionCallParams, output: BackendOutput, *, is_final: bool
    ) -> None:
        """Deliver one output to the frontend.

        Args:
            params: The ``delegate`` call the output belongs to.
            output: The output.
            is_final: Whether it is the backend's final output, which settles the call.
        """
        raise NotImplementedError


class OneShotBackendReplyStrategy(BackendReplyStrategy):
    """Delivers everything the backend produced at once, when it is done.

    The ``delegate`` call's one result carries every output, in order; the
    text alone when there was only one. Reasoning summaries are left out.
    The default for a speech-to-speech frontend, whose function calls accept
    one result. It may not stay the default: if and when those services can
    take intermediate results, progress could reach such a frontend as it
    comes, as :class:`SpeakOnPrefersSpokenBackendReplyStrategy` delivers it.
    """

    def __init__(self):
        """Initialize the strategy."""
        self._progress: dict[str, list[str]] = {}

    async def deliver(
        self, params: FunctionCallParams, output: BackendOutput, *, is_final: bool
    ) -> None:
        """Hold outputs back; deliver them all with the last."""
        if output.is_thought:
            return
        if not is_final:
            self._progress.setdefault(params.tool_call_id, []).append(output.text)
            return
        progress = self._progress.pop(params.tool_call_id, [])
        if progress:
            await params.result_callback({"outputs": [*progress, output.text]})
        else:
            await params.result_callback(output.text)


class SpeakOnPrefersSpokenBackendReplyStrategy(BackendReplyStrategy):
    """Relays the backend's progress as it comes, spoken as the backend's flag says.

    Each output before the final one is recorded as an intermediate tool result;
    the frontend is run on it, and so speaks it, exactly when the output's
    ``prefers_spoken`` flag asks. The default for a text frontend.
    """

    needs_intermediate_results = True

    async def deliver(
        self, params: FunctionCallParams, output: BackendOutput, *, is_final: bool
    ) -> None:
        """Record progress as an intermediate result, run the frontend as flagged."""
        if is_final:
            await params.result_callback(output.text)
            return
        await params.result_callback(
            {"text": output.text},
            properties=FunctionCallResultProperties(is_final=False, run_llm=output.prefers_spoken),
        )


# ---------------------------------------------------------------------------
# The connector
# ---------------------------------------------------------------------------


class BackendConnector:
    """Joins the two layers: the ``delegate`` tool, and what crosses it each way.

    A request strategy defines the tool and composes the request; a reply
    strategy delivers what the backend produces. Either may be given, or left
    to the connector to pick by frontend kind once it is bound:

    +------------------+------------------------------------+--------------------------------------+
    |                  | request                            | reply                                |
    +==================+====================================+======================================+
    | text frontend    | ``TranscriptBackendRequestStrategy`` | ``SpeakOnPrefersSpokenBackendReplyStrategy`` |
    +------------------+------------------------------------+--------------------------------------+
    | realtime frontend| ``ExplicitBackendRequestStrategy``   | ``OneShotBackendReplyStrategy``      |
    +------------------+------------------------------------+--------------------------------------+

    A delegation that fails raises out of the tool handler, which the frontend
    service settles as an error result; one that ends with nothing to say settles
    the call by saying so. The function calls the backend makes on the way are
    reported in the frontend's pipeline as children of the ``delegate`` call,
    for clients to show; nothing else in the pipeline sees them.

    Example::

        connector = BackendConnector(reply_strategy=OneShotBackendReplyStrategy())
    """

    def __init__(
        self,
        *,
        request_strategy: BackendRequestStrategy | None = None,
        reply_strategy: BackendReplyStrategy | None = None,
        timeout_secs: float | None = 120,
    ):
        """Initialize the connector.

        Args:
            request_strategy: How a ``delegate`` call becomes the backend's
                request. Picked by frontend kind when omitted.
            reply_strategy: How the backend's outputs reach the frontend.
                Picked by frontend kind when omitted.
            timeout_secs: How long a delegation may take, including the wait
                for the backend to become ready.
        """
        self._request_strategy = request_strategy
        self._reply_strategy = reply_strategy
        self._timeout_secs = timeout_secs
        self._context: ConnectorContext | None = None
        self._tool: FunctionSchema | None = None

    @property
    def request_strategy(self) -> BackendRequestStrategy:
        """The request strategy in use. Available once bound."""
        assert self._request_strategy is not None, "connector not bound"
        return self._request_strategy

    @property
    def reply_strategy(self) -> BackendReplyStrategy:
        """The reply strategy in use. Available once bound."""
        assert self._reply_strategy is not None, "connector not bound"
        return self._reply_strategy

    @property
    def tool(self) -> FunctionSchema:
        """The ``delegate`` tool to install on the frontend. Available once bound."""
        assert self._tool is not None, "connector not bound"
        return self._tool

    @property
    def frontend_instruction(self) -> str | None:
        """Guidance for the frontend model, from both strategies. Available once bound."""
        parts = [
            self.request_strategy.frontend_instruction,
            self.reply_strategy.frontend_instruction,
        ]
        return "\n\n".join(p for p in parts if p) or None

    def bind(self, context: ConnectorContext) -> None:
        """Settle the strategies for the layers being joined and build the tool.

        Args:
            context: The layers.

        Raises:
            ValueError: If the reply strategy needs intermediate results and the
                frontend is a speech-to-speech service, which cannot take them.
        """
        self._context = context
        if self._request_strategy is None:
            self._request_strategy = (
                ExplicitBackendRequestStrategy()
                if context.frontend_is_realtime
                else TranscriptBackendRequestStrategy()
            )
        if self._reply_strategy is None:
            self._reply_strategy = (
                OneShotBackendReplyStrategy()
                if context.frontend_is_realtime
                else SpeakOnPrefersSpokenBackendReplyStrategy()
            )
        if context.frontend_is_realtime and self._reply_strategy.needs_intermediate_results:
            raise ValueError(
                f"{type(self._reply_strategy).__name__} reports intermediate results, which a "
                "speech-to-speech frontend cannot take: its function calls accept one result"
            )
        self._tool = self.build_tool()

    def build_tool(self) -> FunctionSchema:
        """Build the ``delegate`` tool from the request strategy.

        Override to install a tool of another shape entirely; the service reads
        nothing else from the connector but :attr:`frontend_instruction`.

        Returns:
            The tool, carrying its handler.
        """

        @tool_options(cancel_on_interruption=False)
        async def delegate(params: FunctionCallParams):
            await self.delegate(params)

        return FunctionSchema(
            name=DELEGATE_TOOL_NAME,
            description=self.request_strategy.tool_description(),
            properties=self.request_strategy.tool_parameters,
            required=self.request_strategy.tool_required,
            handler=delegate,
        )

    async def delegate(self, params: FunctionCallParams) -> None:
        """Run one delegation: compose the request, deliver each output.

        Args:
            params: The ``delegate`` call.
        """
        assert self._context is not None, "connector not bound"
        request = await self.request_strategy.compose_request(params)
        logger.debug(f"Delegating to '{self._context.backend_name}': {request!r}")
        finished = False
        async for event in _delegate_to_backend(
            params.pipeline_worker,
            self._context.backend_name,
            request=request,
            timeout_secs=self._timeout_secs,
        ):
            if isinstance(event, BackendToolCall):
                await self.report_tool_call(params, event)
                continue
            if isinstance(event, _BackendFinalOutput):
                finished = True
                await self.reply_strategy.deliver(params, event.output, is_final=True)
            else:
                await self.reply_strategy.deliver(params, event, is_final=False)
        if not finished:
            logger.warning(f"Delegation to '{self._context.backend_name}' produced no final output")
            await params.result_callback({"error": "The backend finished without saying anything."})

    async def report_tool_call(self, params: FunctionCallParams, call: BackendToolCall) -> None:
        """Report a function call the backend made, as the ``delegate`` call's child.

        The call ran in the backend's pipeline; here it is only reported, as
        the :class:`~pipecat.frames.frames.ExternalFunctionCallFrame` for its
        phase, which the RTVI observer turns into function-call events under
        the ``delegate`` call.

        Args:
            params: The ``delegate`` call the backend is working for.
            call: The phase of the backend's call.
        """
        await params.llm.push_frame(call.to_frame(parent_tool_call_id=params.tool_call_id))


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class PipecatDualLLMService(Pipeline):
    """A conversational frontend LLM with a backend it delegates to, in one processor.

    Put it where the LLM goes in a pipeline. It wraps the frontend service,
    installs the connector's ``delegate`` tool on it as a built-in tool,
    appends the connector's guidance to the frontend's system instruction,
    and adds a local backend worker to the pipeline worker so the app never
    wires it up. A built-in tool is sent on every inference beside whatever
    tools the frontend has, and never enters the context's tool set, so a
    tool change announced to the model never mentions it. The tools are the
    backend's; the frontend has ``delegate`` and no more, though a tool it
    must keep, in its context or configured on the service, stays.

    The guidance says when to delegate in general terms. The frontend's own
    system instruction is the place to say what the backend is for, in plain
    words, as the backend's own instruction does: no tool names, nothing to
    update when a tool changes.

    Example::

        llm = PipecatDualLLMService(
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
        """
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
        frontend.register_function(
            DELEGATE_TOOL_NAME, self._connector.tool.handler, cancel_on_interruption=False
        )
        frontend.get_llm_adapter().builtin_tools[DELEGATE_TOOL_NAME] = self._connector.tool
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

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the frontend, and register a local backend worker as a child of the pipeline worker."""
        await super().setup(setup)
        if isinstance(self._backend, BackendLLMWorker):
            await self.pipeline_worker.add_workers(self._backend)

    @property
    def tool(self) -> FunctionSchema:
        """The ``delegate`` tool installed on the frontend."""
        return self._connector.tool
