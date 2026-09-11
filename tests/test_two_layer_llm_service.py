#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for TwoLayerLLMService, BackendConnector and the strategies.

The connector and strategies are exercised directly with a faked delegation
stream. The service is exercised through ``run_test`` with a scripted
backend under a real WorkerRunner, so the delegate tool's whole path runs:
advertised in the context, registered on the frontend, called, delegated
over a job, and answered as tool results.
"""

from unittest.mock import AsyncMock, call

import pytest

from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
)
from pipecat.pipeline import two_layer_llm_service
from pipecat.pipeline.two_layer_llm_service import (
    SILENCE_MARKER,
    AdvisorySpeechFlagBackendReplyStrategy,
    BackendConnector,
    ConnectorContext,
    ExplicitBackendRequestStrategy,
    FinalOnlyBackendReplyStrategy,
    StrictSpeechFlagBackendReplyStrategy,
    TranscriptBackendRequestStrategy,
    TwoLayerLLMService,
    _SilenceFilter,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.workers.llm import BackendLLMWorker, BackendOutput
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
        llm=None,  # type: ignore[arg-type]
        pipeline_worker=None,  # type: ignore[arg-type]
        context=context or LLMContext(),
        result_callback=AsyncMock(),
    )


def _stream(monkeypatch, *outputs: BackendOutput) -> list[dict]:
    """Fake the delegation stream; returns the requests it was given."""
    requests: list[dict] = []

    async def fake(worker, backend_name, *, request, timeout_secs):
        requests.append({"backend": backend_name, "request": request, "timeout": timeout_secs})
        for output in outputs:
            yield output

    monkeypatch.setattr(two_layer_llm_service, "delegate_to_backend", fake)
    return requests


def _bound(connector: BackendConnector, realtime: bool = False) -> BackendConnector:
    connector.bind(ConnectorContext(backend_name="backend", frontend_is_realtime=realtime))
    return connector


# ---------------------------------------------------------------------------
# Connector defaults and the tool it builds
# ---------------------------------------------------------------------------


def test_a_text_frontend_hands_over_the_transcript_and_follows_the_flag():
    connector = _bound(BackendConnector())
    assert isinstance(connector.request, TranscriptBackendRequestStrategy)
    assert isinstance(connector.reply, StrictSpeechFlagBackendReplyStrategy)
    assert connector.tool.name == "delegate"
    assert connector.tool.properties == {}
    assert connector.tool.handler is not None


def test_a_realtime_frontend_words_the_request_and_takes_the_answer_only():
    connector = _bound(BackendConnector(), realtime=True)
    assert isinstance(connector.request, ExplicitBackendRequestStrategy)
    assert isinstance(connector.reply, FinalOnlyBackendReplyStrategy)
    assert connector.tool.required == ["request"]


def test_a_realtime_frontend_refuses_a_reply_strategy_that_streams():
    with pytest.raises(ValueError, match="one result"):
        _bound(BackendConnector(reply=StrictSpeechFlagBackendReplyStrategy()), realtime=True)


def test_the_tool_description_says_what_the_backend_is_for():
    connector = _bound(BackendConnector(backend_description="the weather"))
    assert "for the weather" in connector.tool.description


def test_the_frontend_guidance_comes_from_both_strategies():
    connector = _bound(BackendConnector(reply=AdvisorySpeechFlagBackendReplyStrategy()))
    instruction = connector.frontend_instruction or ""
    assert "Hand off to the backend" in instruction
    assert SILENCE_MARKER in instruction
    assert connector.skip_marker == SILENCE_MARKER


# ---------------------------------------------------------------------------
# Request strategies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_transcript_request_sends_only_what_the_backend_has_not_seen(monkeypatch):
    requests = _stream(monkeypatch, BackendOutput(text="ok", is_final=True))
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
    requests = _stream(monkeypatch, BackendOutput(text="ok", is_final=True))
    connector = _bound(BackendConnector(request=ExplicitBackendRequestStrategy()))

    await connector.delegate(_params(arguments={"request": "Weather in Seattle, Fahrenheit."}))

    assert requests[0]["request"] == "Weather in Seattle, Fahrenheit."


# ---------------------------------------------------------------------------
# Reply strategies
# ---------------------------------------------------------------------------

_PROGRESS = BackendOutput(text="Let me check.", prefers_spoken=False)
_SPOKEN_PROGRESS = BackendOutput(text="Almost there.", prefers_spoken=True)
_ANSWER = BackendOutput(text="It's 62 and raining.", is_final=True)


@pytest.mark.asyncio
async def test_strict_relays_progress_and_runs_the_frontend_as_flagged(monkeypatch):
    _stream(monkeypatch, _PROGRESS, _SPOKEN_PROGRESS, _ANSWER)
    params = _params()

    await _bound(BackendConnector()).delegate(params)

    assert params.result_callback.await_args_list == [  # type: ignore[attr-defined]
        call(
            {"text": "Let me check."},
            properties=FunctionCallResultProperties(is_final=False, run_llm=False),
        ),
        call(
            {"text": "Almost there."},
            properties=FunctionCallResultProperties(is_final=False, run_llm=True),
        ),
        call("It's 62 and raining."),
    ]


@pytest.mark.asyncio
async def test_advisory_passes_the_flag_on_and_always_runs_the_frontend(monkeypatch):
    _stream(monkeypatch, _PROGRESS, _ANSWER)
    params = _params()

    await _bound(BackendConnector(reply=AdvisorySpeechFlagBackendReplyStrategy())).delegate(params)

    assert params.result_callback.await_args_list == [  # type: ignore[attr-defined]
        call(
            {"text": "Let me check.", "prefers_spoken": False},
            properties=FunctionCallResultProperties(is_final=False, run_llm=True),
        ),
        call("It's 62 and raining."),
    ]


@pytest.mark.asyncio
async def test_final_only_drops_progress(monkeypatch):
    _stream(monkeypatch, _PROGRESS, _SPOKEN_PROGRESS, _ANSWER)
    params = _params()

    await _bound(BackendConnector(), realtime=True).delegate(params)

    assert params.result_callback.await_args_list == [call("It's 62 and raining.")]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_delegation_without_an_answer_still_settles_the_call(monkeypatch):
    _stream(monkeypatch, _PROGRESS)
    params = _params()

    await _bound(BackendConnector()).delegate(params)

    assert params.result_callback.await_args_list[-1] == call(  # type: ignore[attr-defined]
        {"error": "The backend finished without an answer."}
    )


# ---------------------------------------------------------------------------
# The silence filter
# ---------------------------------------------------------------------------


async def _texts_through(filter_: _SilenceFilter, *texts: str) -> list[str]:
    frames: list[Frame] = [LLMFullResponseStartFrame()]
    frames += [LLMTextFrame(t) for t in texts]
    frames.append(LLMFullResponseEndFrame())
    down, _ = await run_test(filter_, frames_to_send=frames)
    return [f.text for f in down if isinstance(f, LLMTextFrame)]


@pytest.mark.asyncio
async def test_a_response_that_is_the_marker_says_nothing():
    assert await _texts_through(_SilenceFilter(SILENCE_MARKER), " ", SILENCE_MARKER) == []
    assert await _texts_through(_SilenceFilter(SILENCE_MARKER), f"{SILENCE_MARKER} ok") == []


@pytest.mark.asyncio
async def test_a_response_that_speaks_passes_whole():
    assert await _texts_through(_SilenceFilter(SILENCE_MARKER), " Hel", "lo") == [" Hel", "lo"]


@pytest.mark.asyncio
async def test_the_filter_is_inert_without_a_marker():
    assert await _texts_through(_SilenceFilter(None), SILENCE_MARKER) == [SILENCE_MARKER]


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


def test_the_service_adds_the_connector_guidance_to_the_frontend_prompt():
    frontend = _TextFrontend()
    TwoLayerLLMService(frontend=frontend, backend="backend")
    composed = frontend._settings.system_instruction
    assert composed.startswith("You are a voice assistant.")
    assert "Hand off to the backend" in composed


@pytest.mark.asyncio
async def test_the_service_advertises_the_tool_and_keeps_the_frontend_tools_working():
    frontend = _TextFrontend()
    service = TwoLayerLLMService(frontend=frontend, backend="backend")
    context = LLMContext(tools=[get_current_time])

    await run_test(service, frames_to_send=[LLMContextFrame(context)])

    assert [t.name for t in context.tools.standard_tools] == ["get_current_time", "delegate"]
    assert "delegate" in frontend._functions
    assert "get_current_time" in frontend._functions
    assert not frontend._functions["delegate"].cancel_on_interruption


@pytest.mark.asyncio
async def test_a_tool_change_keeps_the_tool_in_the_frame_and_the_context():
    frontend = _RealtimeFrontend()
    service = TwoLayerLLMService(frontend=frontend, backend="backend")
    context = LLMContext(tools=[get_current_time])
    set_tools = LLMSetToolsFrame(tools=[get_weather])

    await run_test(service, frames_to_send=[LLMContextFrame(context), set_tools])

    assert [t.name for t in set_tools.tools.standard_tools] == ["get_weather", "delegate"]
    assert "delegate" in [t.name for t in context.tools.standard_tools]


@pytest.mark.asyncio
async def test_a_local_backend_answers_through_the_delegate_tool():
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
    service = TwoLayerLLMService(frontend=frontend, backend=backend)

    down, _ = await run_test(
        service,
        frames_to_send=[LLMContextFrame(LLMContext()), SleepFrame(sleep=2.0)],
    )

    results = [f for f in down if isinstance(f, FunctionCallResultFrame)]
    assert [r.result for r in results] == [{"text": "Let me check."}, "It's 62 and raining."]
    assert results[0].properties == FunctionCallResultProperties(is_final=False, run_llm=False)
