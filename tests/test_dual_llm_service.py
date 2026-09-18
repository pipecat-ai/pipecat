#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for PipecatDualLLMService, BackendConnector and the strategies.

The connector and strategies are exercised directly with a faked delegation
stream. The service is exercised through ``run_test`` with a scripted
backend under a real WorkerRunner, so the delegate tool's whole path runs:
advertised in the context, registered on the frontend, called, delegated
over a job, and answered as tool results.
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
    LLMServiceMetadataFrame,
    LLMSetToolsFrame,
)
from pipecat.pipeline import dual_llm_service
from pipecat.pipeline.dual_llm_service import (
    BackendConnector,
    BackendReplyStrategy,
    ConnectorContext,
    ExplicitBackendRequestStrategy,
    OneShotBackendReplyStrategy,
    PipecatDualLLMService,
    SpeakOnPrefersSpokenBackendReplyStrategy,
    TranscriptBackendRequestStrategy,
)
from pipecat.pipeline.job_context import JobError
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.workers.llm import BackendLLMWorker, BackendOutput
from pipecat.workers.llm.backend_llm_worker import BackendToolCall, _BackendFinalOutput
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


def _stream(
    monkeypatch, *outputs: BackendOutput | BackendToolCall | _BackendFinalOutput
) -> list[dict]:
    """Fake the delegation stream; returns the requests it was given."""
    requests: list[dict] = []

    async def fake(worker, backend_name, *, request, timeout_secs):
        requests.append({"backend": backend_name, "request": request, "timeout": timeout_secs})
        for output in outputs:
            yield output

    monkeypatch.setattr(dual_llm_service, "_delegate_to_backend", fake)
    return requests


def _bound(connector: BackendConnector, realtime: bool = False) -> BackendConnector:
    connector.bind(ConnectorContext(backend_name="backend", frontend_is_realtime=realtime))
    return connector


# ---------------------------------------------------------------------------
# Connector defaults and the tool it builds
# ---------------------------------------------------------------------------


def test_a_text_frontend_hands_over_the_transcript_and_follows_the_flag():
    connector = _bound(BackendConnector())
    assert isinstance(connector.request_strategy, TranscriptBackendRequestStrategy)
    assert isinstance(connector.reply_strategy, SpeakOnPrefersSpokenBackendReplyStrategy)
    assert connector.tool.name == "delegate"
    assert connector.tool.properties == {}
    assert connector.tool.handler is not None


def test_a_realtime_frontend_words_the_request_and_takes_every_output_at_once():
    connector = _bound(BackendConnector(), realtime=True)
    assert isinstance(connector.request_strategy, ExplicitBackendRequestStrategy)
    assert isinstance(connector.reply_strategy, OneShotBackendReplyStrategy)
    assert connector.tool.required == ["request"]


def test_a_realtime_frontend_refuses_a_reply_strategy_that_streams():
    with pytest.raises(ValueError, match="one result"):
        _bound(
            BackendConnector(reply_strategy=SpeakOnPrefersSpokenBackendReplyStrategy()),
            realtime=True,
        )


def test_the_tool_description_frames_a_handoff():
    connector = _bound(BackendConnector())
    assert "One handoff per reply" in connector.tool.description


def test_the_frontend_guidance_comes_from_the_strategies():
    connector = _bound(BackendConnector())
    assert "delegate tool" in (connector.frontend_instruction or "")


# ---------------------------------------------------------------------------
# Request strategies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_transcript_request_sends_only_what_the_backend_has_not_seen(monkeypatch):
    requests = _stream(monkeypatch, _BackendFinalOutput(BackendOutput(text="ok")))
    connector = _bound(BackendConnector(timeout_secs=7))
    context = LLMContext([{"role": "user", "content": "weather in seattle?"}])

    await connector.delegate(_params(context))
    context.add_message({"role": "assistant", "content": "It's raining."})
    context.add_message({"role": "user", "content": "and boston?"})
    await connector.delegate(_params(context))

    assert requests[0]["backend"] == "backend"
    assert requests[0]["timeout"] == 7
    assert requests[0]["request"].startswith(
        "Voice conversation so far:\nUSER: weather in seattle?\n"
    )
    assert requests[1]["request"].startswith(
        "Voice conversation since the previous delegation:\n"
        "ASSISTANT: It's raining.\nUSER: and boston?\n"
    )


@pytest.mark.asyncio
async def test_the_explicit_request_sends_the_model_words(monkeypatch):
    requests = _stream(monkeypatch, _BackendFinalOutput(BackendOutput(text="ok")))
    connector = _bound(BackendConnector(request_strategy=ExplicitBackendRequestStrategy()))

    await connector.delegate(_params(arguments={"request": "Weather in Seattle, Fahrenheit."}))

    assert requests[0]["request"] == "Weather in Seattle, Fahrenheit."


# ---------------------------------------------------------------------------
# Reply strategies
# ---------------------------------------------------------------------------

_PROGRESS = BackendOutput(text="Let me check.", prefers_spoken=False)
_SPOKEN_PROGRESS = BackendOutput(text="Almost there.", prefers_spoken=True)
_FINAL = _BackendFinalOutput(BackendOutput(text="It's 62 and raining."))
_THOUGHT = BackendOutput(text="Weather first.", is_thought=True, prefers_spoken=False)
_EMPTY_FINAL = _BackendFinalOutput(BackendOutput(text=""))


@pytest.mark.asyncio
async def test_speak_on_prefers_spoken_relays_progress_and_runs_the_frontend_as_flagged(
    monkeypatch,
):
    _stream(monkeypatch, _PROGRESS, _THOUGHT, _SPOKEN_PROGRESS, _FINAL)
    params = _params()

    await _bound(BackendConnector()).delegate(params)

    assert params.result_callback.await_args_list == [  # type: ignore[attr-defined]
        call(
            {"text": "Let me check."},
            properties=FunctionCallResultProperties(is_final=False, run_llm=False),
        ),
        call(
            {"reasoning": "Weather first."},
            properties=FunctionCallResultProperties(is_final=False, run_llm=False),
        ),
        call(
            {"text": "Almost there."},
            properties=FunctionCallResultProperties(is_final=False, run_llm=True),
        ),
        call("It's 62 and raining."),
    ]


@pytest.mark.asyncio
async def test_one_shot_delivers_every_output_together(monkeypatch):
    _stream(monkeypatch, _PROGRESS, _THOUGHT, _SPOKEN_PROGRESS, _FINAL)
    params = _params()

    await _bound(BackendConnector(), realtime=True).delegate(params)

    assert params.result_callback.await_args_list == [  # type: ignore[attr-defined]
        call({"outputs": ["Let me check.", "Almost there.", "It's 62 and raining."]})
    ]


@pytest.mark.asyncio
async def test_one_shot_delivers_what_it_held_when_the_final_output_is_empty(monkeypatch):
    _stream(monkeypatch, _PROGRESS, _SPOKEN_PROGRESS, _EMPTY_FINAL)
    params = _params()
    connector = _bound(BackendConnector(), realtime=True)

    await connector.delegate(params)

    assert params.result_callback.await_args_list == [  # type: ignore[attr-defined]
        call({"outputs": ["Let me check.", "Almost there."]})
    ]
    assert connector.reply_strategy._progress == {}  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_one_shot_drops_what_it_held_when_the_delegation_fails(monkeypatch):
    async def fake(worker, backend_name, *, request, timeout_secs):
        yield _PROGRESS
        raise JobError("backend errored")

    monkeypatch.setattr(dual_llm_service, "_delegate_to_backend", fake)
    params = _params()
    connector = _bound(BackendConnector(), realtime=True)

    with pytest.raises(JobError):
        await connector.delegate(params)

    params.result_callback.assert_not_awaited()  # type: ignore[attr-defined]
    assert connector.reply_strategy._progress == {}  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_one_shot_delivers_a_lone_output_as_text(monkeypatch):
    _stream(monkeypatch, _FINAL)
    params = _params()

    await _bound(BackendConnector(), realtime=True).delegate(params)

    assert params.result_callback.await_args_list == [call("It's 62 and raining.")]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_delivery_that_fails_closes_the_stream(monkeypatch):
    """Closing the stream is what cancels the backend's job, so it must happen at once."""
    closed = False

    async def fake(worker, backend_name, *, request, timeout_secs):
        nonlocal closed
        try:
            yield _PROGRESS
            yield _FINAL
        finally:
            closed = True

    monkeypatch.setattr(dual_llm_service, "_delegate_to_backend", fake)

    class _Broken(BackendReplyStrategy):
        async def deliver(self, params, output, *, is_final):
            raise RuntimeError("frontend went away")

    with pytest.raises(RuntimeError):
        await _bound(BackendConnector(reply_strategy=_Broken())).delegate(_params())

    assert closed


@pytest.mark.asyncio
async def test_the_backends_calls_are_reported_as_children_of_the_delegate_call(monkeypatch):
    _stream(
        monkeypatch,
        BackendToolCall("in_progress", "get_weather", "toolu_1", arguments={"location": "Seattle"}),
        _FINAL,
    )
    params = _params()

    await _bound(BackendConnector()).delegate(params)

    (pushed,) = [c.args[0] for c in params.llm.push_frame.await_args_list]
    assert isinstance(pushed, ExternalFunctionCallInProgressFrame)
    assert (pushed.function_name, pushed.tool_call_id) == ("get_weather", "toolu_1")
    assert pushed.parent_tool_call_id == "call_1"
    # The call is reported, not delivered to the frontend as a result.
    assert params.result_callback.await_args_list == [call("It's 62 and raining.")]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_an_output_with_no_text_is_skipped_by_both_strategies(monkeypatch):
    blank = BackendOutput(text="", prefers_spoken=True)
    for realtime in (False, True):
        _stream(monkeypatch, blank, _PROGRESS, blank, _FINAL)
        params = _params()

        await _bound(BackendConnector(), realtime=realtime).delegate(params)

        results = [c.args[0] for c in params.result_callback.await_args_list]  # type: ignore[attr-defined]
        assert "" not in results
        assert results[-1] in (
            "It's 62 and raining.",
            {"outputs": ["Let me check.", "It's 62 and raining."]},
        )


@pytest.mark.asyncio
async def test_a_delegation_with_nothing_to_say_still_settles_the_call(monkeypatch):
    _stream(monkeypatch, _PROGRESS, _EMPTY_FINAL)
    params = _params()

    await _bound(BackendConnector()).delegate(params)

    assert params.result_callback.await_args_list[-1] == call(  # type: ignore[attr-defined]
        {"error": "The backend finished without saying anything."}
    )


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


def test_the_service_adds_the_connector_guidance_to_the_frontend_prompt():
    frontend = _TextFrontend()
    PipecatDualLLMService(frontend=frontend, backend="backend")
    composed = frontend._settings.system_instruction
    assert composed.startswith("You are a voice assistant.")
    assert "delegate tool" in composed


def _tool_names(converted) -> list[str]:
    return [t.get("name") or t["function"]["name"] for t in converted]


@pytest.mark.asyncio
async def test_delegate_is_a_built_in_tool_beside_the_frontends_own():
    frontend = _TextFrontend()
    service = PipecatDualLLMService(frontend=frontend, backend="backend")
    context = LLMContext(tools=[get_current_time])
    set_tools = LLMSetToolsFrame(tools=[get_weather])

    await run_test(service, frames_to_send=[LLMContextFrame(context), set_tools])

    # Never in the context's or a tool change's tool set, so no diff sees it.
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
    assert not frontend._functions["delegate"].cancel_on_interruption


@pytest.mark.asyncio
async def test_a_local_backend_is_heard_through_the_delegate_tool():
    """The whole path, with the backend worker added by the service itself."""
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
    service = PipecatDualLLMService(frontend=frontend, backend=backend)

    down, _ = await run_test(
        service,
        frames_to_send=[LLMContextFrame(LLMContext()), SleepFrame(sleep=2.0)],
    )

    results = [f for f in down if isinstance(f, FunctionCallResultFrame)]
    assert [r.result for r in results] == [{"text": "Let me check."}, "It's 62 and raining."]
    assert results[0].properties == FunctionCallResultProperties(is_final=False, run_llm=False)
    # The backend's own call reached the frontend's pipeline as a report only.
    reported = [f for f in down if isinstance(f, ExternalFunctionCallFrame)]
    assert [(type(f), f.function_name) for f in reported] == [
        (ExternalFunctionCallStartedFrame, "get_weather"),
        (ExternalFunctionCallInProgressFrame, "get_weather"),
        (ExternalFunctionCallResultFrame, "get_weather"),
    ]
    assert {f.parent_tool_call_id for f in reported} == {"call_1"}
