#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A two-layer LLM: a conversational frontend delegating to a backend.

The frontend is any LLM service, text or speech-to-speech, and holds the
conversation. The backend is a :class:`~pipecat.workers.llm.backend_llm_worker.BackendLLMWorker`
running a heavier model with the tools, and does the work the frontend hands
off. :class:`TwoLayerLLMService` wraps the frontend so the pair drops into a
pipeline where an LLM goes, and installs the ``delegate`` tool that joins
them.

How a delegation crosses is the :class:`BackendConnector`'s business, built
from two strategies: a :class:`BackendRequestStrategy` defines the tool the
frontend calls and turns a call into the request the backend receives, and a
:class:`BackendReplyStrategy` turns each thing the backend produces into what
the frontend hears about it. The connector picks defaults by frontend kind:
a text frontend hands over the conversation and relays the backend's
progress; a speech-to-speech frontend words the request itself and takes only
the answer, since its function calls accept one result.
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultProperties,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.utils.types import NotGiven, is_given
from pipecat.workers.llm.backend_llm_worker import (
    DEFAULT_TRANSCRIPT_INSTRUCTION,
    BackendLLMWorker,
    BackendOutput,
    delegate_to_backend,
    render_transcript_request,
)

#: Name of the tool the frontend calls to delegate.
DELEGATE_TOOL_NAME = "delegate"

#: What the ``delegate`` tool says the backend is for, unless the app says.
DEFAULT_BACKEND_DESCRIPTION = "anything needing tools, current information or careful reasoning"

#: What a frontend answers, alone, when it decides to say nothing.
SILENCE_MARKER = "∅"


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

    def tool_description(self, backend_description: str) -> str:
        """Describe the ``delegate`` tool to the frontend model.

        Args:
            backend_description: What the backend is for, in the app's words.

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
    :func:`~pipecat.workers.llm.backend_llm_worker.render_transcript_request`.
    The default for a text frontend, whose context is current when the tool
    runs.
    """

    frontend_instruction = (
        "Hand off to the backend as soon as you know a request is for it: it reads "
        "the conversation itself, so you need not word the request. While it works, "
        "keep the conversation going. When its result comes back, relay it in your "
        "own words."
    )

    def __init__(self, *, instruction: str = DEFAULT_TRANSCRIPT_INSTRUCTION):
        """Initialize the strategy.

        Args:
            instruction: What the backend should do with the transcript, placed
                after it.
        """
        self._instruction = instruction
        self._delegated_through = 0

    def tool_description(self, backend_description: str) -> str:
        """Describe the tool: hand the conversation over, for what the backend is for."""
        return (
            f"Hand the conversation to the backend, for {backend_description}. The backend "
            "reads the conversation itself, so there is nothing to pass."
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
        return render_transcript_request(conversation, instruction=self._instruction, first=first)


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
    frontend_instruction = (
        "When you hand off to the backend, word the request so it stands on its own: "
        "the user's goal, the exact details they gave and their latest correction. "
        "While it works, keep the conversation going. When its result comes back, "
        "relay it in your own words."
    )

    def tool_description(self, backend_description: str) -> str:
        """Describe the tool: hand a worded request over, for what the backend is for."""
        return f"Hand a request to the backend, for {backend_description}."

    async def compose_request(self, params: FunctionCallParams) -> str:
        """Send the model's request as it stands."""
        return str(params.arguments.get("request") or "")


# ---------------------------------------------------------------------------
# Reply strategies: backend → frontend
# ---------------------------------------------------------------------------


class BackendReplyStrategy:
    """Turns each :class:`BackendOutput` into what the frontend hears about it.

    The final output settles the ``delegate`` call as its result; what
    happens to the outputs before it is what strategies differ on.
    """

    #: Whether the strategy reports outputs before the final one as
    #: intermediate tool results. A speech-to-speech frontend cannot take
    #: those: its function calls accept one result.
    needs_intermediate_results: bool = False
    #: Guidance appended to the frontend's system instruction, if any.
    frontend_instruction: str | None = None
    #: A marker the frontend answers with, alone, to say nothing; ``None``
    #: when the strategy gives it no such choice.
    skip_marker: str | None = None

    async def deliver(self, params: FunctionCallParams, output: BackendOutput) -> None:
        """Deliver one output to the frontend.

        Args:
            params: The ``delegate`` call the output belongs to.
            output: The output.
        """
        if output.is_final:
            await params.result_callback(output.text)


class FinalOnlyBackendReplyStrategy(BackendReplyStrategy):
    """Delivers the backend's answer and nothing before it.

    The default for a speech-to-speech frontend, whose function calls accept
    one result.
    """


class StrictSpeechFlagBackendReplyStrategy(BackendReplyStrategy):
    """Relays the backend's progress, spoken or not as the backend's flag says.

    Each output before the answer is recorded as an intermediate tool result;
    the frontend is run on it, and so speaks it, exactly when the output's
    ``prefers_spoken`` flag asks. The default for a text frontend.
    """

    needs_intermediate_results = True

    async def deliver(self, params: FunctionCallParams, output: BackendOutput) -> None:
        """Record progress as an intermediate result, run the frontend as flagged."""
        if output.is_final:
            await params.result_callback(output.text)
            return
        await params.result_callback(
            {"text": output.text},
            properties=FunctionCallResultProperties(is_final=False, run_llm=output.prefers_spoken),
        )


class AdvisorySpeechFlagBackendReplyStrategy(BackendReplyStrategy):
    """Relays the backend's progress and lets the frontend decide what to say.

    Each output before the answer is recorded as an intermediate tool result
    carrying the backend's ``prefers_spoken`` flag, and the frontend is run on
    every one. The frontend model weighs the flag against the conversation and
    either speaks or answers with :data:`SILENCE_MARKER` alone, which
    :class:`TwoLayerLLMService` drops so nothing is said. Text frontends only.
    """

    needs_intermediate_results = True
    skip_marker = SILENCE_MARKER
    frontend_instruction = (
        "Partial results from the backend carry a prefers_spoken flag: whether the backend "
        "would like the user to hear that text now. It is advice, not an order. Speak a "
        f"partial result when it helps the conversation; when it does not, reply with {SILENCE_MARKER} "
        "and nothing else, and nothing will be said. The final result is always for the user."
    )

    async def deliver(self, params: FunctionCallParams, output: BackendOutput) -> None:
        """Record progress with its flag and run the frontend on it."""
        if output.is_final:
            await params.result_callback(output.text)
            return
        await params.result_callback(
            {"text": output.text, "prefers_spoken": output.prefers_spoken},
            properties=FunctionCallResultProperties(is_final=False, run_llm=True),
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
    | text frontend    | ``TranscriptBackendRequestStrategy`` | ``StrictSpeechFlagBackendReplyStrategy`` |
    +------------------+------------------------------------+--------------------------------------+
    | realtime frontend| ``ExplicitBackendRequestStrategy``   | ``FinalOnlyBackendReplyStrategy``      |
    +------------------+------------------------------------+--------------------------------------+

    A delegation that fails raises out of the tool handler, which the frontend
    service settles as an error result; one that ends without an answer settles
    the call by saying so.

    Example::

        connector = BackendConnector(
            reply=AdvisorySpeechFlagBackendReplyStrategy(),
            backend_description="current information such as the weather",
        )
    """

    def __init__(
        self,
        *,
        request: BackendRequestStrategy | None = None,
        reply: BackendReplyStrategy | None = None,
        backend_description: str = DEFAULT_BACKEND_DESCRIPTION,
        timeout_secs: float | None = 120,
    ):
        """Initialize the connector.

        Args:
            request: How a ``delegate`` call becomes the backend's request.
                Picked by frontend kind when omitted.
            reply: How the backend's outputs reach the frontend. Picked by
                frontend kind when omitted.
            backend_description: What the backend is for, as the ``delegate``
                tool's description tells the frontend model.
            timeout_secs: How long a delegation may take, including the wait
                for the backend to become ready.
        """
        self._request = request
        self._reply = reply
        self._backend_description = backend_description
        self._timeout_secs = timeout_secs
        self._context: ConnectorContext | None = None
        self._tool: FunctionSchema | None = None

    @property
    def request(self) -> BackendRequestStrategy:
        """The request strategy in use. Available once bound."""
        assert self._request is not None, "connector not bound"
        return self._request

    @property
    def reply(self) -> BackendReplyStrategy:
        """The reply strategy in use. Available once bound."""
        assert self._reply is not None, "connector not bound"
        return self._reply

    @property
    def tool(self) -> FunctionSchema:
        """The ``delegate`` tool to install on the frontend. Available once bound."""
        assert self._tool is not None, "connector not bound"
        return self._tool

    @property
    def frontend_instruction(self) -> str | None:
        """Guidance for the frontend model, from both strategies. Available once bound."""
        parts = [self.request.frontend_instruction, self.reply.frontend_instruction]
        return "\n\n".join(p for p in parts if p) or None

    @property
    def skip_marker(self) -> str | None:
        """The reply strategy's silence marker, if it gives the frontend one."""
        return self.reply.skip_marker

    def bind(self, context: ConnectorContext) -> None:
        """Settle the strategies for the layers being joined and build the tool.

        Args:
            context: The layers.

        Raises:
            ValueError: If the reply strategy needs intermediate results and the
                frontend is a speech-to-speech service, which cannot take them.
        """
        self._context = context
        if self._request is None:
            self._request = (
                ExplicitBackendRequestStrategy()
                if context.frontend_is_realtime
                else TranscriptBackendRequestStrategy()
            )
        if self._reply is None:
            self._reply = (
                FinalOnlyBackendReplyStrategy()
                if context.frontend_is_realtime
                else StrictSpeechFlagBackendReplyStrategy()
            )
        if context.frontend_is_realtime and self._reply.needs_intermediate_results:
            raise ValueError(
                f"{type(self._reply).__name__} reports intermediate results, which a "
                "speech-to-speech frontend cannot take: its function calls accept one result"
            )
        self._tool = self.build_tool()

    def build_tool(self) -> FunctionSchema:
        """Build the ``delegate`` tool from the request strategy.

        Override to install a tool of another shape entirely; the service reads
        nothing else from the connector but :attr:`frontend_instruction` and
        :attr:`skip_marker`.

        Returns:
            The tool, carrying its handler.
        """

        @tool_options(cancel_on_interruption=False)
        async def delegate(params: FunctionCallParams):
            await self.delegate(params)

        return FunctionSchema(
            name=DELEGATE_TOOL_NAME,
            description=self.request.tool_description(self._backend_description),
            properties=self.request.tool_parameters,
            required=self.request.tool_required,
            handler=delegate,
        )

    async def delegate(self, params: FunctionCallParams) -> None:
        """Run one delegation: compose the request, deliver each output.

        Args:
            params: The ``delegate`` call.
        """
        assert self._context is not None, "connector not bound"
        request = await self.request.compose_request(params)
        logger.debug(f"Delegating to '{self._context.backend_name}': {request!r}")
        answered = False
        async for output in delegate_to_backend(
            params.pipeline_worker,
            self._context.backend_name,
            request=request,
            timeout_secs=self._timeout_secs,
        ):
            answered = answered or output.is_final
            await self.reply.deliver(params, output)
        if not answered:
            logger.warning(f"Delegation to '{self._context.backend_name}' produced no answer")
            await params.result_callback({"error": "The backend finished without an answer."})


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class _SilenceFilter(FrameProcessor):
    """Drops a frontend response that opens with the silence marker.

    Text is held back until the response's first non-blank character says
    whether the marker is coming; a response that opens with it loses its
    text, so the TTS says nothing and the assistant aggregator records no
    turn. Inert without a marker.
    """

    def __init__(self, marker: str | None, **kwargs):
        super().__init__(**kwargs)
        self._marker = marker
        self._buffer = ""
        self._skipping: bool | None = None  # None: undecided for this response

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._marker is None or direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
            self._reset()
        elif isinstance(frame, LLMTextFrame):
            await self._handle_text(frame)
            return
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._skipping is None and self._buffer.strip():
                # Whitespace, then a marker prefix that never completed.
                await self.push_frame(LLMTextFrame(self._buffer))
            self._reset()
        await self.push_frame(frame, direction)

    async def _handle_text(self, frame: LLMTextFrame):
        if self._skipping is True:
            return
        if self._skipping is False:
            await self.push_frame(frame)
            return
        assert self._marker is not None
        self._buffer += frame.text
        opening = self._buffer.lstrip()
        if opening.startswith(self._marker):
            self._skipping = True
        elif not opening or self._marker.startswith(opening):
            return  # nothing decisive yet
        else:
            self._skipping = False
        if self._skipping:
            logger.debug(f"{self}: the frontend chose to say nothing")
        else:
            await self.push_frame(LLMTextFrame(self._buffer))
        self._buffer = ""

    def _reset(self):
        self._buffer = ""
        self._skipping = None


def _with_tool(tools: ToolsSchema | NotGiven | None, tool: FunctionSchema) -> ToolsSchema | None:
    """Return ``tools`` with ``tool`` added, or ``None`` if it is already there."""
    if tools is None or not is_given(tools):
        return ToolsSchema(standard_tools=[tool])
    if any(schema.name == tool.name for schema in tools.standard_tools):
        return None
    # Direct functions go back in as the callables they came from, so their
    # handlers still register; the rest of the standard tools are schemas.
    direct = {wrapper.name: wrapper.function for wrapper in tools.direct_functions}
    standard: list[Any] = [direct.get(schema.name, schema) for schema in tools.standard_tools] + [
        tool
    ]
    return ToolsSchema(standard_tools=standard, custom_tools=tools.custom_tools)


class TwoLayerLLMService(Pipeline):
    """A conversational frontend LLM with a backend it delegates to, in one processor.

    Put it where the LLM goes in a pipeline. It wraps the frontend service,
    installs the connector's ``delegate`` tool on it (adding the tool to the
    context's tools as they pass, and to any tool change), appends the
    connector's guidance to the frontend's system instruction, and adds a local
    backend worker to the pipeline worker so the app never wires it up.

    Example::

        llm = TwoLayerLLMService(
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
        frontend: LLMService,
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
        self._filter = _SilenceFilter(self._connector.skip_marker)
        self._context: LLMContext | None = None
        super().__init__([frontend, self._filter])

    @property
    def frontend(self) -> LLMService:
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, keeping the ``delegate`` tool among the advertised tools.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        if isinstance(frame, LLMContextFrame):
            self._context = frame.context
            self._advertise_in(frame.context)
        elif isinstance(frame, LLMSetToolsFrame):
            # The aggregator upstream has already set these on the context; a
            # speech-to-speech frontend syncs its handlers from the frame and
            # its session from the context, so the tool goes in both.
            tools = _with_tool(LLMContext._normalize_and_validate_tools(frame.tools), self.tool)
            if tools is not None:
                frame.tools = tools
            if self._context is not None:
                self._advertise_in(self._context)
        await super().process_frame(frame, direction)

    @property
    def tool(self) -> FunctionSchema:
        """The ``delegate`` tool installed on the frontend."""
        return self._connector.tool

    def _advertise_in(self, context: LLMContext) -> None:
        tools = _with_tool(context.tools, self.tool)
        if tools is not None:
            context.set_tools(tools)
