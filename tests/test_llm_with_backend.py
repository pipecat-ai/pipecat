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

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from pipecat.frames.frames import (
    ExternalFunctionCallFrame,
    ExternalFunctionCallInProgressFrame,
    ExternalFunctionCallResultFrame,
    ExternalFunctionCallStartedFrame,
    Frame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMServiceMetadataFrame,
    LLMSetToolsFrame,
)
from pipecat.pipeline.llm_with_backend import (
    BackendConnector,
    ConnectorContext,
    ExplicitBackendRequestStrategy,
    LLMWithBackend,
    TranscriptBackendRequestStrategy,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
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

    async def cancel(self, reason: str) -> bool:
        self.reasons.append(reason)
        return self._cancelled


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
    assert [t.name for t in connector.tools] == ["delegate", "cancel_delegated_work"]
    assert connector.tool.properties == {}
    assert all(t.handler is not None for t in connector.tools)


def test_a_realtime_frontend_words_the_request():
    connector = _bound(realtime=True)
    assert isinstance(connector.request_strategy, ExplicitBackendRequestStrategy)
    assert connector.tool.required == ["request"]


def test_the_tool_descriptions_frame_a_handoff_and_a_stop():
    connector = _bound()
    assert "One handoff per reply" in connector.tool.description
    assert "Stop all the work" in connector.tools[1].description


def test_the_frontend_guidance_covers_delegation_and_the_backends_messages():
    guidance = _bound().frontend_instruction or ""
    assert "delegate tool" in guidance
    assert 'marked "Backend:"' in guidance
    assert '"Backend (working):"' in guidance
    assert "cancel_delegated_work" in guidance


# ---------------------------------------------------------------------------
# Request strategies
# ---------------------------------------------------------------------------


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
        "Act on the user's most recent request in the conversation above, and report its result as soon as you have it, before going on with other work."
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
        "Report the result of this request as soon as you have it, before going on with other work."
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


@pytest.mark.asyncio
async def test_cancel_reports_whether_there_was_work_to_stop():
    for cancelled, status in ((True, "cancelled"), (False, "nothing_running")):
        session = _FakeSession(cancelled=cancelled)
        params = _params()

        await _bound(session=session).cancel(params)

        assert session.reasons == ["cancelled by the user"]
        assert params.result_callback.await_args_list == [  # type: ignore[attr-defined]
            call({"status": status}, properties=FunctionCallResultProperties(run_llm=True))
        ]


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
        ([{"role": "developer", "content": "Backend: Let me check."}], False),
        ([{"role": "developer", "content": "Backend (thinking): Weather first."}], False),
        ([{"role": "developer", "content": "Backend: It's 62 and raining."}], True),
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
    assert pushed.parent_tool_call_id is None
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
        "cancel_delegated_work",
    ]
    assert _tool_names(adapter.from_standard_tools(NOT_GIVEN)) == [
        "delegate",
        "cancel_delegated_work",
    ]
    assert frontend.has_function("delegate")
    assert frontend.has_function("cancel_delegated_work")


@pytest.mark.asyncio
async def test_a_local_backend_is_heard_through_the_frontends_conversation():
    """The whole path, with the backend worker added and attached by the service itself."""
    backend = BackendLLMWorker(
        name="backend",
        llm=_ScriptedLLM(
            [
                [("text", "Let me check."), ("call", "get_weather", "c1", {"location": "Seattle"})],
                [("text", "It's 62 and raining.")],
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
        ("Backend: Let me check.", False),
        ("Backend (working): get_weather(location='Seattle')", False),
        ("Backend: It's 62 and raining.", True),
    ]
    # The backend's own call reached the frontend's pipeline as a report only.
    reported = [f for f in down if isinstance(f, ExternalFunctionCallFrame)]
    assert [(type(f), f.function_name) for f in reported] == [
        (ExternalFunctionCallStartedFrame, "get_weather"),
        (ExternalFunctionCallInProgressFrame, "get_weather"),
        (ExternalFunctionCallResultFrame, "get_weather"),
    ]
    assert {f.parent_tool_call_id for f in reported} == {None}
