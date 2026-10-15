#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LLMWithBackend, BackendConnector and the request strategies.

The connector is exercised directly with a faked session. The service is
exercised through ``run_test`` with a scripted backend under a real
WorkerRunner, so the whole path runs: the tools installed on the frontend,
``delegate`` called, the message sent over the session, and the backend's
outputs appended to the frontend's conversation.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import pytest

from pipecat.classifiers.base_classifier import BaseClassifier, ClassifierError, YesNoResult
from pipecat.frames.frames import (
    ExternalFunctionCallInProgressFrame,
    ExternalFunctionCallResultFrame,
    ExternalFunctionCallsStartedFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMServiceMetadataFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    MetricsFrame,
    StartFrame,
    TextFrame,
)
from pipecat.pipeline.llm_with_backend import (
    RELAY_NOTE,
    TRANSCRIPT_FRONTEND_INSTRUCTION,
    BackendConnector,
    ConnectorContext,
    ExplicitBackendRequestStrategy,
    LLMWithBackend,
    PredictiveDelegation,
    TranscriptBackendRequestStrategy,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm import BackendLLMWorker, BackendOutput
from pipecat.workers.llm.backend_llm_worker import BackendError, BackendIdle, BackendToolCall
from tests.test_backend_llm_worker import _ScriptedLLM, get_weather


def _settings(system_instruction: str | None = None) -> LLMSettings:
    return LLMSettings(
        model="test-model",
        system_instruction=system_instruction,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        filter_incomplete_user_turns=None,
        user_turn_completion_config=None,
    )


class _TextFrontend(LLMService):
    """A text frontend that forwards every frame."""

    def __init__(self, **kwargs):
        super().__init__(settings=_settings("You are a voice assistant."), **kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _RealtimeFrontend(_TextFrontend):
    """A frontend that announces itself as speech-to-speech."""

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        return LLMServiceMetadataFrame(service_name=self.name, is_realtime_service=True)


class _DelegatingFrontend(_TextFrontend):
    """A text frontend that calls ``delegate`` on its first context frame."""

    def __init__(self, arguments: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self._arguments = arguments or {}
        self._delegated = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and not self._delegated:
            self._delegated = True
            await self.push_frame(LLMFullResponseStartFrame())
            await self.run_function_calls(
                [
                    FunctionCallFromLLM(
                        context=frame.context,
                        tool_call_id="call_1",
                        function_name="delegate",
                        arguments=self._arguments,
                    )
                ]
            )
            await self.push_frame(LLMFullResponseEndFrame())


async def get_current_time(params: FunctionCallParams):
    """Get the current time."""
    await params.result_callback({"time": "noon"})


def _params(context: LLMContext | None = None, arguments: dict | None = None) -> FunctionCallParams:
    return FunctionCallParams(
        function_name="delegate",
        tool_call_id="call_1",
        arguments=arguments or {},
        llm=SimpleNamespace(push_frame=AsyncMock()),  # type: ignore[arg-type]
        pipeline_worker=None,  # type: ignore[arg-type]
        context=context or LLMContext(),
        result_callback=AsyncMock(),
    )


class _FakeSession:
    """Stands in for the session with the backend."""

    def __init__(self, status: str = "idle", cancelled: bool = True):
        self.requests: list[str] = []
        self.reasons: list[str] = []
        self._status = status
        self._cancelled = cancelled

    async def send(self, request: str) -> str:
        self.requests.append(request)
        return self._status


def _bound(
    connector: BackendConnector | None = None,
    *,
    realtime: bool = False,
    session: _FakeSession | None = None,
) -> BackendConnector:
    connector = connector or BackendConnector()
    connector.bind(ConnectorContext(backend_name="backend", frontend_is_realtime=realtime))
    if session is not None:
        connector._session = session  # type: ignore[assignment]
        connector._session_open.set()
    return connector


# ---------------------------------------------------------------------------
# Connector defaults and the tools it builds
# ---------------------------------------------------------------------------


def test_a_text_frontend_hands_over_the_transcript():
    connector = _bound()
    assert isinstance(connector.request_strategy, TranscriptBackendRequestStrategy)
    assert [t.name for t in connector.tools] == ["delegate"]
    assert connector.tool.properties == {}
    assert all(t.handler is not None for t in connector.tools)


def test_a_realtime_frontend_words_the_request():
    connector = _bound(realtime=True)
    assert isinstance(connector.request_strategy, ExplicitBackendRequestStrategy)
    assert connector.tool.required == ["request"]


def test_a_realtime_frontend_refuses_the_transcript_strategy():
    connector = BackendConnector(request_strategy=TranscriptBackendRequestStrategy())
    with pytest.raises(ValueError, match="TranscriptBackendRequestStrategy"):
        _bound(connector, realtime=True)


def test_a_realtime_frontend_takes_a_given_explicit_strategy():
    strategy = ExplicitBackendRequestStrategy()
    connector = _bound(BackendConnector(request_strategy=strategy), realtime=True)
    assert connector.request_strategy is strategy


def test_the_tool_description_frames_a_handoff():
    connector = _bound()
    assert "One handoff per reply" in connector.tool.description


def test_the_frontend_guidance_covers_delegation_and_the_backends_messages():
    guidance = _bound().frontend_instruction or ""
    assert "delegate tool" in guidance
    assert 'marked "Backend:"' in guidance
    assert '"Backend (working):"' in guidance
    assert "to start, change or stop its work" in guidance


def test_the_strategys_guidance_sits_inside_the_delegating_paragraph():
    guidance = _bound().frontend_instruction or ""
    assert guidance.startswith("THE BACKEND:")
    delegating = guidance.split("\n\n")[0]
    assert delegating.endswith(" " + TRANSCRIPT_FRONTEND_INSTRUCTION)
    assert "{strategy}" not in guidance
    assert guidance.index("ONE ASSISTANT:") < guidance.index("AFTER DELEGATING:")
    assert guidance.index("AFTER DELEGATING:") < guidance.index("BACKEND MESSAGES:")
    # The instruction ends by restating its rule for messages about delegated work.
    assert guidance.split("\n\n")[-1].startswith(
        "Whatever else a reply does, a user message about work"
    )


def test_an_app_can_replace_the_frontends_pairing_instruction():
    placed = _bound(
        BackendConnector(frontend_pairing_instruction="Hand over: {strategy} Then talk.")
    )
    assert (
        placed.frontend_instruction
        == "Hand over: " + TRANSCRIPT_FRONTEND_INSTRUCTION + " Then talk."
    )
    appended = _bound(BackendConnector(frontend_pairing_instruction="Hand things over."))
    assert (
        appended.frontend_instruction == "Hand things over.\n\n" + TRANSCRIPT_FRONTEND_INSTRUCTION
    )


@pytest.mark.asyncio
async def test_a_strategy_takes_its_three_texts_as_arguments():
    strategy = ExplicitBackendRequestStrategy(
        frontend_instruction="Say it plainly.",
        tool_description="Hand it over.",
        request_instruction="Do it now.",
    )
    connector = _bound(BackendConnector(request_strategy=strategy))
    assert "waiting. Say it plainly." in (connector.frontend_instruction or "")
    assert connector.tool.description == "Hand it over."
    request = await strategy.compose_request(_params(arguments={"request": "Book a taxi"}))
    assert request == "Book a taxi\n\nDo it now."


# ---------------------------------------------------------------------------
# Request strategies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_transcript_request_skips_messages_in_a_services_own_format():
    """A Google frontend records its calls as LLMSpecificMessage; the transcript leaves them out."""
    session = _FakeSession()
    connector = _bound(session=session)
    context = LLMContext(
        [
            {"role": "user", "content": "weather in seattle?"},
            LLMSpecificMessage(llm="google", message={"parts": [{"text": "call"}]}),
            {"role": "assistant", "content": "Let me check."},
        ]
    )

    await connector.delegate(_params(context))

    assert session.requests == [
        "Voice conversation so far:\n"
        "USER: weather in seattle?\n"
        "ASSISTANT: Let me check.\n"
        "\n"
        "Act on the user's most recent request in the conversation above. If it asks for something, tell the user the result as soon as you have it, before going on with other work, in a message that begins with >>. If it only stops or changes work already under way, tell them nothing: the assistant already has."
    ]


@pytest.mark.asyncio
async def test_the_transcript_request_sends_only_what_the_backend_has_not_seen():
    session = _FakeSession()
    connector = _bound(session=session)
    context = LLMContext([{"role": "user", "content": "weather in seattle?"}])

    await connector.delegate(_params(context))
    context.add_message({"role": "assistant", "content": "It's raining."})
    context.add_message({"role": "developer", "content": "Backend: Looking it up."})
    context.add_message({"role": "user", "content": "and boston?"})
    await connector.delegate(_params(context))

    assert session.requests[0].startswith("Voice conversation so far:\nUSER: weather in seattle?\n")
    # The backend's own messages are not sent back to it.
    assert session.requests[1] == (
        "Voice conversation since the previous delegation:\n"
        "ASSISTANT: It's raining.\n"
        "USER: and boston?\n"
        "\n"
        "Act on the user's most recent request in the conversation above. If it asks for something, tell the user the result as soon as you have it, before going on with other work, in a message that begins with >>. If it only stops or changes work already under way, tell them nothing: the assistant already has."
    )


@pytest.mark.asyncio
async def test_a_delegate_call_with_nothing_new_sends_nothing():
    """A second call for the same turn finds the transcript slice empty and settles quietly."""
    session = _FakeSession()
    connector = _bound(session=session)
    context = LLMContext([{"role": "user", "content": "weather in seattle?"}])

    await connector.delegate(_params(context))
    params = _params(context)
    await connector.delegate(params)

    assert session.requests == [session.requests[0]]
    assert params.result_callback.await_args_list == [  # type: ignore[attr-defined]
        call(
            {"status": "already_delegated"}, properties=FunctionCallResultProperties(run_llm=False)
        )
    ]


@pytest.mark.asyncio
async def test_the_explicit_request_sends_the_model_words():
    session = _FakeSession()
    connector = _bound(
        BackendConnector(request_strategy=ExplicitBackendRequestStrategy()), session=session
    )

    await connector.delegate(_params(arguments={"request": "Weather in Seattle, Fahrenheit."}))

    assert session.requests == [
        "Weather in Seattle, Fahrenheit.\n\n"
        "If this request asks for something, tell the user the result as soon as you have it, before going on with other work, in a message that begins with >>. If it only stops or changes work already under way, tell them nothing: the assistant already has."
    ]


# ---------------------------------------------------------------------------
# The delegate and cancel tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_settles_at_once_with_what_the_backend_was_doing():
    for status in ("idle", "working"):
        params = _params(LLMContext([{"role": "user", "content": "do it"}]))

        await _bound(session=_FakeSession(status=status)).delegate(params)

        assert params.result_callback.await_args_list == [  # type: ignore[attr-defined]
            call(
                {"status": "delegated", "backend": status},
                properties=FunctionCallResultProperties(run_llm=True),
            )
        ]


@pytest.mark.asyncio
async def test_delegate_can_leave_the_frontend_quiet():
    params = _params(LLMContext([{"role": "user", "content": "do it"}]))

    await _bound(BackendConnector(respond_on_delegate=False), session=_FakeSession()).delegate(
        params
    )

    (settled,) = params.result_callback.await_args_list  # type: ignore[attr-defined]
    assert settled.kwargs["properties"] == FunctionCallResultProperties(run_llm=False)


@pytest.mark.asyncio
async def test_delegate_fails_when_the_backend_never_attaches():
    connector = _bound(BackendConnector(timeout_secs=0.05))
    params = _params(LLMContext([{"role": "user", "content": "do it"}]))

    with pytest.raises(RuntimeError, match="not attached"):
        await connector.delegate(params)

    params.result_callback.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outputs_are_appended_to_the_frontend_and_run_it_as_flagged():
    connector = _bound()
    frontend = SimpleNamespace(queue_frame=AsyncMock(), push_frame=AsyncMock())

    await connector.deliver(frontend, BackendOutput(text="Let me check.", prefers_spoken=False))  # type: ignore[arg-type]
    await connector.deliver(
        frontend, BackendOutput(text="Weather first.", is_thought=True, prefers_spoken=False)
    )  # type: ignore[arg-type]
    await connector.deliver(frontend, BackendOutput(text="It's 62 and raining."))  # type: ignore[arg-type]
    await connector.deliver(frontend, BackendOutput(text=""))  # type: ignore[arg-type]

    appended = [c.args[0] for c in frontend.queue_frame.await_args_list]
    assert [(f.messages, f.run_llm) for f in appended] == [
        ([{"role": "developer", "content": "Backend (note): Let me check."}], False),
        ([{"role": "developer", "content": "Backend (thinking): Weather first."}], False),
        (
            [{"role": "developer", "content": f"Backend: It's 62 and raining.\n\n{RELAY_NOTE}"}],
            True,
        ),
    ]
    frontend.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_backends_calls_are_reported_and_nothing_else_happens_to_them():
    connector = _bound()
    frontend = SimpleNamespace(queue_frame=AsyncMock(), push_frame=AsyncMock())

    await connector.deliver(
        frontend,  # type: ignore[arg-type]
        BackendToolCall("in_progress", "get_weather", "toolu_1", arguments={"location": "Seattle"}),
    )

    (pushed,) = [c.args[0] for c in frontend.push_frame.await_args_list]
    assert isinstance(pushed, ExternalFunctionCallInProgressFrame)
    assert (pushed.function_name, pushed.tool_call_id) == ("get_weather", "toolu_1")
    # The call is also recorded in the conversation, silently, so the frontend
    # can say what the backend is doing.
    (appended,) = [c.args[0] for c in frontend.queue_frame.await_args_list]
    assert appended.run_llm is False
    assert appended.messages == [
        {"role": "developer", "content": "Backend (working): get_weather(location='Seattle')"}
    ]

    frontend = SimpleNamespace(queue_frame=AsyncMock(), push_frame=AsyncMock())
    await connector.deliver(
        frontend, BackendToolCall("result", "get_weather", "toolu_1", result={})
    )  # type: ignore[arg-type]
    frontend.queue_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_backend_error_is_spoken_and_idle_is_not():
    connector = _bound()
    frontend = SimpleNamespace(queue_frame=AsyncMock(), push_frame=AsyncMock())

    await connector.deliver(frontend, BackendIdle())  # type: ignore[arg-type]
    await connector.deliver(frontend, BackendError(error="provider down"))  # type: ignore[arg-type]

    (appended,) = [c.args[0] for c in frontend.queue_frame.await_args_list]
    assert appended.run_llm is True
    assert appended.messages[0]["content"].startswith("Backend: The work could not be completed")


@pytest.mark.asyncio
async def test_render_output_is_the_seam_for_another_wording():
    class _Terse(BackendConnector):
        def render_output(self, output):
            return None if output.is_thought else {"role": "user", "content": f"[be] {output.text}"}

    connector = _bound(_Terse())
    frontend = SimpleNamespace(queue_frame=AsyncMock(), push_frame=AsyncMock())

    await connector.deliver(frontend, BackendOutput(text="thinking", is_thought=True))  # type: ignore[arg-type]
    await connector.deliver(frontend, BackendOutput(text="done"))  # type: ignore[arg-type]

    (appended,) = [c.args[0] for c in frontend.queue_frame.await_args_list]
    assert appended.messages == [{"role": "user", "content": "[be] done"}]


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


def test_a_frontend_that_declines_the_role_is_refused():
    class _Declining(_TextFrontend):
        def llm_with_backend_role_objection(self, role):
            return f"cannot be the {role}: it has a backend of its own"

    with pytest.raises(ValueError, match="cannot be the frontend"):
        LLMWithBackend(frontend=_Declining(), backend="backend")


def test_the_service_adds_the_connector_guidance_to_the_frontend_prompt():
    frontend = _TextFrontend()
    LLMWithBackend(frontend=frontend, backend="backend")
    composed = frontend._settings.system_instruction
    assert composed.startswith("You are a voice assistant.")
    assert "delegate tool" in composed


def _tool_names(converted) -> list[str]:
    return [t.get("name") or t["function"]["name"] for t in converted]


@pytest.mark.asyncio
async def test_the_tools_are_built_in_beside_the_frontends_own():
    frontend = _TextFrontend()
    service = LLMWithBackend(frontend=frontend, backend="backend")
    context = LLMContext(tools=[get_current_time])
    set_tools = LLMSetToolsFrame(tools=[get_weather])

    await run_test(service, frames_to_send=[LLMContextFrame(context), set_tools])

    # Never in the context's or a tool change's tool set, so no diff sees them.
    assert [t.name for t in context.tools.standard_tools] == ["get_current_time"]
    assert set_tools.tools == [get_weather]
    # Sent on every inference all the same, beside whatever tools there are.
    adapter = frontend.get_llm_adapter()
    assert _tool_names(adapter.from_standard_tools(context.tools)) == [
        "get_current_time",
        "delegate",
    ]
    assert _tool_names(adapter.from_standard_tools(NOT_GIVEN)) == ["delegate"]
    assert frontend.has_function("delegate")


@pytest.mark.asyncio
async def test_client_tracing_sends_each_exchange_as_a_server_message():
    backend = BackendLLMWorker(
        name="backend",
        llm=_ScriptedLLM(
            [
                [("call", "get_weather", "c1", {"location": "Seattle"})],
                [("text", ">> It's 62 and raining.")],
            ]
        ),
        context=LLMContext(tools=[get_weather]),
    )
    service = LLMWithBackend(
        frontend=_DelegatingFrontend(),
        backend=backend,
        connector=BackendConnector(client_trace=True),
    )

    down, _ = await run_test(
        service,
        frames_to_send=[
            LLMContextFrame(LLMContext([{"role": "user", "content": "weather in seattle?"}])),
            SleepFrame(sleep=2.0),
        ],
    )

    traced = [f.data for f in down if isinstance(f, RTVIServerMessageFrame)]
    assert all(d["type"] == "llm-with-backend" for d in traced)
    assert [d["event"] for d in traced] == [
        "request",
        "tool_call",
        "tool_call",
        "tool_call",
        "output",
        "idle",
    ]
    assert traced[0]["backend"] == "idle" and "weather in seattle?" in traced[0]["text"]
    assert traced[-2]["text"] == "It's 62 and raining." and traced[-2]["spoken"] is True


@pytest.mark.asyncio
async def test_a_local_backend_is_heard_through_the_frontends_conversation():
    """The whole path, with the backend worker added and attached by the service itself."""
    backend = BackendLLMWorker(
        name="backend",
        llm=_ScriptedLLM(
            [
                [("text", "Let me check."), ("call", "get_weather", "c1", {"location": "Seattle"})],
                [("text", ">> It's 62 and raining.")],
            ]
        ),
        context=LLMContext(tools=[get_weather]),
    )
    frontend = _DelegatingFrontend()
    service = LLMWithBackend(frontend=frontend, backend=backend)

    down, _ = await run_test(
        service,
        frames_to_send=[
            LLMContextFrame(LLMContext([{"role": "user", "content": "weather in seattle?"}])),
            SleepFrame(sleep=2.0),
        ],
    )

    (result,) = [f for f in down if isinstance(f, FunctionCallResultFrame)]
    assert result.result == {"status": "delegated", "backend": "idle"}
    assert result.properties == FunctionCallResultProperties(run_llm=True)
    appended = [f for f in down if isinstance(f, LLMMessagesAppendFrame)]
    assert [(f.messages[0]["content"], f.run_llm) for f in appended] == [
        ("Backend (note): Let me check.", False),
        ("Backend (working): get_weather(location='Seattle')", False),
        (f"Backend: It's 62 and raining.\n\n{RELAY_NOTE}", True),
    ]
    # A message queued while the bot speaks waits behind the speech; an
    # interruption must not drop it on the way to the aggregator.
    assert all(not f.interruptible for f in appended)
    # The backend's own call reached the frontend's pipeline as a report only.
    (started,) = [f for f in down if isinstance(f, ExternalFunctionCallsStartedFrame)]
    assert [c.function_name for c in started.function_calls] == ["get_weather"]
    reported = [
        f
        for f in down
        if isinstance(f, (ExternalFunctionCallInProgressFrame, ExternalFunctionCallResultFrame))
    ]
    assert [(type(f), f.function_name) for f in reported] == [
        (ExternalFunctionCallInProgressFrame, "get_weather"),
        (ExternalFunctionCallResultFrame, "get_weather"),
    ]


@pytest.mark.asyncio
async def test_a_local_backend_is_added_before_the_frontend_starts():
    """The frontend can delegate as soon as it gets the StartFrame, so the
    backend worker is added before the frontend starts."""
    events = []

    class _StartRecordingFrontend(_DelegatingFrontend):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            if isinstance(frame, StartFrame):
                events.append("frontend_started")
            await super().process_frame(frame, direction)

    add_workers = BaseWorker.add_workers

    async def slow_add_workers(worker, *workers, **kwargs):
        await asyncio.sleep(0.2)
        await add_workers(worker, *workers, **kwargs)
        events.append("backend_added")

    backend = BackendLLMWorker(
        name="backend", llm=_ScriptedLLM([[("text", "Done.")]]), context=LLMContext()
    )
    service = LLMWithBackend(frontend=_StartRecordingFrontend(), backend=backend)

    with patch.object(BaseWorker, "add_workers", slow_add_workers):
        await run_test(service, frames_to_send=[SleepFrame(sleep=0.5)])

    assert events[:2] == ["backend_added", "frontend_started"]


# ---------------------------------------------------------------------------
# Predictive delegation
# ---------------------------------------------------------------------------


class _StreamingFrontend(LLMService):
    """A text frontend whose inference waits before its first output, then streams.

    Its output is two text chunks ``streaming`` seconds apart, or a call to
    ``call`` with no text. It records whether an inference was cancelled.
    """

    def __init__(
        self,
        *,
        first_output_after: float,
        streaming: float = 0.0,
        call: str | None = None,
        system_instruction: str = "You are a voice assistant.",
    ):
        super().__init__(settings=_settings(system_instruction))
        self._first_output_after = first_output_after
        self._streaming = streaming
        self._call = call
        self.cancelled = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await asyncio.sleep(self._first_output_after)
            if self._call:
                await self.run_function_calls(
                    [
                        FunctionCallFromLLM(
                            context=frame.context,
                            tool_call_id="model_call",
                            function_name=self._call,
                            arguments={},
                        )
                    ]
                )
            else:
                await self.push_frame(LLMTextFrame("Hello"))
                await asyncio.sleep(self._streaming)
                await self.push_frame(LLMTextFrame(" there"))
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            await self.push_frame(LLMFullResponseEndFrame())


class _ScriptedClassifier(BaseClassifier):
    """Answers after ``after`` seconds, "yes" for the user messages in ``yes``.

    Records each state it was asked about.
    """

    def __init__(self, *, after: float, yes: tuple[str, ...] = (), fail: bool = False):
        super().__init__()
        self._after = after
        self._yes = yes
        self._fail = fail
        self.states: list = []

    async def _ask(self, state, questions, images=()):
        self.states.append(state)
        await asyncio.sleep(self._after)
        if self._fail:
            raise ClassifierError("unavailable")
        last = state["conversation"][-1]["text"]
        probability = 0.95 if last in self._yes else 0.05
        return {name: YesNoResult(probability=probability) for name in questions}, None


class _RecordingConnector(BackendConnector):
    """A connector whose ``delegate`` records each call instead of reaching a backend."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delegated: list[str] = []

    async def delegate(self, params: FunctionCallParams) -> None:
        self.delegated.append(params.tool_call_id)
        await params.result_callback(
            {"status": "delegated"}, properties=FunctionCallResultProperties(run_llm=False)
        )


def _user_turn(text: str = "Fix the flaky test.") -> LLMContextFrame:
    return LLMContextFrame(LLMContext([{"role": "user", "content": text}]))


def _with_predictive_delegation(frontend: LLMService, classifier: BaseClassifier):
    connector = _RecordingConnector()
    service = LLMWithBackend(
        frontend=frontend,
        backend="backend",
        connector=connector,
        predictive_delegation=PredictiveDelegation(classifier),
    )
    return service, connector


def _texts(frames) -> list[str]:
    return [f.text for f in frames if isinstance(f, LLMTextFrame)]


def _early_delegations(delegated: list[str]) -> list[str]:
    return [d for d in delegated if d != "model_call"]


@pytest.mark.asyncio
async def test_a_yes_before_the_frontends_first_output_delegates_the_turn_early():
    frontend = _StreamingFrontend(first_output_after=0.4)
    classifier = _ScriptedClassifier(after=0.05, yes=("Fix the flaky test.",))
    service, connector = _with_predictive_delegation(frontend, classifier)

    down, _ = await run_test(
        service,
        frames_to_send=[
            _user_turn(),
            TextFrame(text="queued behind the inference"),
            SleepFrame(sleep=0.6),
        ],
    )

    assert len(_early_delegations(connector.delegated)) == 1
    assert frontend.cancelled
    assert _texts(down) == []
    # What was queued behind the cancelled inference is still processed.
    assert [f.text for f in down if type(f) is TextFrame] == ["queued behind the inference"]
    # Downstream, the call sits inside the cancelled inference's response, as
    # a model's own call would.
    shape = [
        type(f)
        for f in down
        if isinstance(
            f, (LLMFullResponseStartFrame, FunctionCallInProgressFrame, LLMFullResponseEndFrame)
        )
    ]
    assert shape == [
        LLMFullResponseStartFrame,
        FunctionCallInProgressFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_a_yes_after_the_frontends_first_text_leaves_the_turn_to_the_frontend():
    frontend = _StreamingFrontend(first_output_after=0.05, streaming=0.3)
    classifier = _ScriptedClassifier(after=0.15, yes=("Fix the flaky test.",))
    service, connector = _with_predictive_delegation(frontend, classifier)

    down, _ = await run_test(service, frames_to_send=[_user_turn(), SleepFrame(sleep=0.5)])

    assert connector.delegated == []
    assert _texts(down) == ["Hello", " there"]


@pytest.mark.asyncio
async def test_a_yes_after_the_frontends_own_delegate_call_adds_nothing():
    frontend = _StreamingFrontend(first_output_after=0.05, call="delegate")
    classifier = _ScriptedClassifier(after=0.15, yes=("Fix the flaky test.",))
    service, connector = _with_predictive_delegation(frontend, classifier)

    await run_test(service, frames_to_send=[_user_turn(), SleepFrame(sleep=0.3)])

    assert connector.delegated == ["model_call"]


@pytest.mark.asyncio
async def test_a_no_leaves_the_turn_to_the_frontend():
    frontend = _StreamingFrontend(first_output_after=0.2)
    classifier = _ScriptedClassifier(after=0.05)
    service, connector = _with_predictive_delegation(frontend, classifier)

    down, _ = await run_test(service, frames_to_send=[_user_turn(), SleepFrame(sleep=0.3)])

    assert connector.delegated == []
    assert _texts(down) == ["Hello", " there"]


@pytest.mark.asyncio
async def test_a_failing_classifier_leaves_the_turn_to_the_frontend():
    frontend = _StreamingFrontend(first_output_after=0.2)
    classifier = _ScriptedClassifier(after=0.05, fail=True)
    service, connector = _with_predictive_delegation(frontend, classifier)

    down, _ = await run_test(service, frames_to_send=[_user_turn(), SleepFrame(sleep=0.3)])

    assert connector.delegated == []
    assert _texts(down) == ["Hello", " there"]


@pytest.mark.asyncio
async def test_a_yes_for_a_turn_still_queued_is_carried_out_once_it_starts():
    frontend = _StreamingFrontend(first_output_after=0.3)
    classifier = _ScriptedClassifier(after=0.05, yes=("Fix the flaky test.",))
    service, connector = _with_predictive_delegation(frontend, classifier)

    down, _ = await run_test(
        service,
        frames_to_send=[
            _user_turn("Tell me a joke."),
            _user_turn("Fix the flaky test."),
            SleepFrame(sleep=0.8),
        ],
    )

    # The earlier response ran to the end; the queued turn was delegated
    # early once its inference started.
    assert _texts(down) == ["Hello", " there"]
    assert len(_early_delegations(connector.delegated)) == 1
    assert frontend.cancelled


@pytest.mark.asyncio
async def test_an_interruption_leaves_the_turn_to_the_frontend():
    frontend = _StreamingFrontend(first_output_after=0.3)
    classifier = _ScriptedClassifier(after=0.2, yes=("Fix the flaky test.",))
    service, connector = _with_predictive_delegation(frontend, classifier)

    await run_test(
        service,
        frames_to_send=[_user_turn(), SleepFrame(sleep=0.1), InterruptionFrame(), SleepFrame(0.4)],
    )

    assert connector.delegated == []


@pytest.mark.asyncio
async def test_only_a_new_user_message_is_raced():
    frontend = _StreamingFrontend(first_output_after=0.0)
    classifier = _ScriptedClassifier(after=0.0)
    service, _ = _with_predictive_delegation(frontend, classifier)
    turn = _user_turn()
    developer = LLMContextFrame(LLMContext([{"role": "developer", "content": "Greet the user."}]))
    speculative = LLMContextFrame(
        LLMContext([{"role": "user", "content": "Fix the"}]), speculation=True
    )

    await run_test(
        service,
        frames_to_send=[developer, speculative, turn, SleepFrame(0.1), turn, SleepFrame(0.1)],
    )

    # The developer message and the speculative turn aren't raced, and a turn
    # raced once isn't raced again.
    assert len(classifier.states) == 1


@pytest.mark.asyncio
async def test_no_turn_is_raced_while_the_frontend_filters_incomplete_turns():
    frontend = _StreamingFrontend(first_output_after=0.0)
    classifier = _ScriptedClassifier(after=0.0)
    service, _ = _with_predictive_delegation(frontend, classifier)

    await run_test(
        service,
        frames_to_send=[
            LLMUpdateSettingsFrame(delta=LLMSettings(filter_incomplete_user_turns=True)),
            SleepFrame(0.1),
            _user_turn(),
            SleepFrame(0.1),
        ],
    )

    assert classifier.states == []


@pytest.mark.asyncio
async def test_the_classifier_sees_the_frontends_instruction_and_the_latest_conversation():
    frontend = _StreamingFrontend(first_output_after=0.0, system_instruction="You are Pip.")
    classifier = _ScriptedClassifier(after=0.0)
    service, _ = _with_predictive_delegation(frontend, classifier)
    turn = LLMContextFrame(
        LLMContext(
            [
                {"role": "user", "content": "Fix the flaky test."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "delegate", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"status": "delegated"}'},
                {"role": "developer", "content": f"Backend: Fixed it.\n\n{RELAY_NOTE}"},
                {"role": "user", "content": "Thanks!"},
            ]
        )
    )

    await run_test(service, frames_to_send=[turn, SleepFrame(0.1)])

    (state,) = classifier.states
    # What the frontend's model reads: the app's instruction, then the pairing
    # instruction appended to it.
    assert state["assistant"].startswith("You are Pip.\n\nTHE BACKEND:")
    assert state["conversation"] == [
        {"speaker": "user", "text": "Fix the flaky test."},
        {"speaker": "assistant", "action": "handed the request to the backend"},
        {"speaker": "backend", "text": "Backend: Fixed it."},
        {"speaker": "user", "text": "Thanks!"},
    ]


@pytest.mark.asyncio
async def test_the_classifiers_metrics_reach_the_pipeline():
    frontend = _StreamingFrontend(first_output_after=0.0)
    classifier = _ScriptedClassifier(after=0.0)
    service, _ = _with_predictive_delegation(frontend, classifier)

    down, _ = await run_test(service, frames_to_send=[_user_turn(), SleepFrame(0.1)])

    assert any(isinstance(f, MetricsFrame) for f in down)


@pytest.mark.asyncio
@pytest.mark.parametrize("respond_on_delegate", [True, False])
async def test_an_early_delegation_is_settled_like_the_models_own(respond_on_delegate):
    backend = BackendLLMWorker(
        name="backend",
        llm=_ScriptedLLM([[("text", ">> On it.")]]),
        context=LLMContext(),
    )
    frontend = _StreamingFrontend(first_output_after=1.0)
    service = LLMWithBackend(
        frontend=frontend,
        backend=backend,
        connector=BackendConnector(respond_on_delegate=respond_on_delegate),
        predictive_delegation=PredictiveDelegation(
            _ScriptedClassifier(after=0.05, yes=("Fix the flaky test.",))
        ),
    )

    down, _ = await run_test(service, frames_to_send=[_user_turn(), SleepFrame(sleep=1.5)])

    (result,) = [f for f in down if isinstance(f, FunctionCallResultFrame)]
    assert result.result["status"] == "delegated"
    assert result.properties == FunctionCallResultProperties(run_llm=respond_on_delegate)


def test_predictive_delegation_needs_a_text_frontend():
    with pytest.raises(ValueError, match="text frontend"):
        LLMWithBackend(
            frontend=_RealtimeFrontend(),
            backend="backend",
            connector=BackendConnector(request_strategy=ExplicitBackendRequestStrategy()),
            predictive_delegation=PredictiveDelegation(_ScriptedClassifier(after=0.0)),
        )


def test_predictive_delegation_needs_a_delegate_call_with_no_arguments():
    with pytest.raises(ValueError, match="ExplicitBackendRequestStrategy"):
        LLMWithBackend(
            frontend=_TextFrontend(),
            backend="backend",
            connector=BackendConnector(request_strategy=ExplicitBackendRequestStrategy()),
            predictive_delegation=PredictiveDelegation(_ScriptedClassifier(after=0.0)),
        )
