#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for OpenAILiveLLMService: session configuration, event → frame
mapping, and the Responses-delegation function call path.

Server events are scripted through the receive handler; outgoing client events
are recorded from ``send_client_event``. ``__init__`` does no I/O.
"""

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.live import events
from pipecat.services.openai.live import llm as live_llm
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _weather_tool() -> FunctionSchema:
    return FunctionSchema(
        name="get_weather",
        description="Get the weather.",
        properties={"location": {"type": "string"}},
        required=["location"],
    )


def _responses_delegation(**settings) -> live_llm.ResponsesDelegation:
    return OpenAILiveLLMService.ResponsesDelegation(
        settings=OpenAIResponsesLLMService.Settings(model="gpt-5.4-mini", **settings)
    )


def _make_service(*, delegation=None, settings=None) -> OpenAILiveLLMService:
    return OpenAILiveLLMService(api_key="test-key", settings=settings, delegation=delegation)


class _FakeWebSocket:
    """Minimal async-iterable websocket that yields scripted JSON strings."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


class _FrameRecorder:
    """Records frames passed to a service's ``push_frame``."""

    def __init__(self):
        self.frames: list[tuple[Any, FrameDirection]] = []

    async def __call__(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.frames.append((frame, direction))

    def of_types(self, *types) -> list[Any]:
        return [f for f, _ in self.frames if isinstance(f, types)]


class _EventRecorder:
    """Records outgoing client events (as dicts) sent via ``send_client_event``."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: events.ClientEvent):
        self.events.append(event.model_dump(exclude_none=True))

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]


async def _drive(service: OpenAILiveLLMService, scripted: list[dict[str, Any]]) -> None:
    """Feed scripted server-event dicts through the receive handler."""
    service._websocket = _FakeWebSocket([json.dumps(e) for e in scripted])
    await service._receive_task_handler()


def _turn_created(turn_id: str, role: str, transcript: str = "") -> dict[str, Any]:
    return {
        "type": "turn.created",
        "turn": {
            "id": turn_id,
            "role": role,
            "start_ms": 0,
            "end_ms": 100,
            "transcript": transcript,
        },
    }


def _turn_delta(turn_id: str, delta: str) -> dict[str, Any]:
    return {
        "type": "turn.delta",
        "turn_id": turn_id,
        "delta": delta,
        "start_ms": 100,
        "end_ms": 200,
    }


def _turn_done(turn_id: str, role: str, transcript: str) -> dict[str, Any]:
    return {
        "type": "turn.done",
        "turn": {
            "id": turn_id,
            "role": role,
            "start_ms": 0,
            "end_ms": 200,
            "transcript": transcript,
        },
    }


def _session_started() -> dict[str, Any]:
    return {
        "type": "session.started",
        "session": {"id": "rtc_123", "model": "gpt-live-1-marble-alpha"},
    }


# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_session_update_client_mode():
    """The first context configures instructions, voice, client delegation and initial items."""
    service = _make_service(settings=OpenAILiveLLMService.Settings(voice="cedar"))
    recorder = _EventRecorder()
    service.send_client_event = recorder

    context = LLMContext(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "developer", "content": "Greet the user."},
            {"role": "user", "content": [{"type": "text", "text": "hi there"}]},
            {"role": "assistant", "content": "Hello!"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 62}'},
        ]
    )
    await service._handle_context(context)

    (update,) = recorder.of_type("session.update")
    session = update["session"]
    assert session["instructions"] == "Be brief."
    assert session["audio"] == {"output": {"voice": "cedar"}}
    assert session["delegation"] == {"type": "client"}
    assert session["initial_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Greet the user."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi there"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello!"}],
        },
    ]


@pytest.mark.asyncio
async def test_initial_session_update_responses_mode():
    """Responses settings map onto delegation.responses; tools/tool_choice come from the context."""
    delegation = OpenAILiveLLMService.ResponsesDelegation(
        settings=OpenAIResponsesLLMService.Settings(
            model="gpt-5.4-mini",
            system_instruction="You are the backend.",
            reasoning=OpenAIResponsesLLMService.ReasoningConfig(effort="low"),
            max_completion_tokens=256,
            temperature=0.5,
            extra={"text": {"verbosity": "low"}},
        ),
        service_tier="priority",
    )
    service = _make_service(delegation=delegation)
    recorder = _EventRecorder()
    service.send_client_event = recorder

    context = LLMContext(
        [{"role": "system", "content": "Be brief."}],
        tools=[_weather_tool()],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    await service._handle_context(context)

    (update,) = recorder.of_type("session.update")
    session = update["session"]
    assert "initial_items" not in session
    responses = session["delegation"]["responses"]
    assert session["delegation"]["type"] == "responses"
    assert responses["model"] == "gpt-5.4-mini"
    assert responses["instructions"] == "You are the backend."
    assert responses["reasoning"] == {"effort": "low"}
    assert responses["max_output_tokens"] == 256
    assert responses["service_tier"] == "priority"
    assert responses["text"] == {"verbosity": "low"}
    assert "temperature" not in responses
    assert responses["tool_choice"] == {"type": "function", "name": "get_weather"}
    (tool,) = responses["tools"]
    assert tool["type"] == "function"
    assert tool["name"] == "get_weather"
    assert "strict" not in tool


def test_responses_delegation_requires_model():
    with pytest.raises(ValueError, match="model is required"):
        _make_service(
            delegation=OpenAILiveLLMService.ResponsesDelegation(
                settings=OpenAIResponsesLLMService.Settings()
            )
        )


@pytest.mark.asyncio
async def test_system_instruction_setting_wins_over_context_system_message():
    service = _make_service(
        settings=OpenAILiveLLMService.Settings(system_instruction="From settings.")
    )
    recorder = _EventRecorder()
    service.send_client_event = recorder

    await service._handle_context(LLMContext([{"role": "system", "content": "From context."}]))

    (update,) = recorder.of_type("session.update")
    assert update["session"]["instructions"] == "From settings."


@pytest.mark.asyncio
async def test_initial_items_keep_the_most_recent_128():
    service = _make_service()
    recorder = _EventRecorder()
    service.send_client_event = recorder

    messages = [{"role": "user", "content": f"message {i}"} for i in range(200)]
    await service._handle_context(LLMContext(messages))

    (update,) = recorder.of_type("session.update")
    items = update["session"]["initial_items"]
    assert len(items) == events.MAX_INITIAL_ITEMS
    assert items[0]["content"][0]["text"] == "message 72"
    assert items[-1]["content"][0]["text"] == "message 199"


@pytest.mark.asyncio
async def test_later_context_frames_send_nothing_in_client_mode():
    service = _make_service()
    recorder = _EventRecorder()
    service.send_client_event = recorder

    context = LLMContext([{"role": "user", "content": "hi"}])
    await service._handle_context(context)
    service._session_started = True
    context.add_message({"role": "developer", "content": "Appended by the app."})
    await service._handle_context(context)

    assert len(recorder.events) == 1


@pytest.mark.asyncio
async def test_tools_change_sends_sparse_update_in_responses_mode():
    service = _make_service(delegation=_responses_delegation())
    recorder = _EventRecorder()
    service.send_client_event = recorder

    context = LLMContext([{"role": "user", "content": "hi"}], tools=[_weather_tool()])
    await service._handle_context(context)
    service._session_started = True

    # Same tools: nothing to send.
    await service._handle_context(context)
    assert len(recorder.of_type("session.update")) == 1

    # New tool set: sparse delegation update carrying only the tools.
    context.set_tools(ToolsSchema(standard_tools=[]))
    await service._handle_context(context)
    updates = recorder.of_type("session.update")
    assert len(updates) == 2
    assert updates[1]["session"] == {
        "delegation": {"type": "responses", "responses": {"tools": []}}
    }


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_audio_delta_pushes_24khz_tts_audio_frame():
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder
    audio = b"\x01\x02" * 240

    await _drive(
        service,
        [
            {
                "type": "output_audio.delta",
                "audio": base64.b64encode(audio).decode("ascii"),
                "start_ms": 0,
                "end_ms": 10,
            }
        ],
    )

    (frame,) = recorder.of_types(TTSAudioRawFrame)
    assert frame.audio == audio
    assert frame.sample_rate == 24000
    assert frame.num_channels == 1


@pytest.mark.asyncio
async def test_input_audio_is_dropped_until_session_started_and_resampled_to_24khz():
    service = _make_service()
    recorder = _EventRecorder()
    service.send_client_event = recorder
    audio_24k = b"\x00\x01" * 480

    await service._send_user_audio(InputAudioRawFrame(audio_24k, 24000, 1))
    assert recorder.events == []

    service._session_started = True
    await service._send_user_audio(InputAudioRawFrame(audio_24k, 24000, 1))
    (append,) = recorder.of_type("input_audio.append")
    assert base64.b64decode(append["audio"]) == audio_24k

    # 16 kHz input is resampled to 24 kHz. The stream resampler emits in its
    # own chunk sizes with some latency, so feed 200 ms and check the total.
    for _ in range(10):
        await service._send_user_audio(InputAudioRawFrame(b"\x00\x01" * 320, 16000, 1))
    resampled = b"".join(
        base64.b64decode(e["audio"]) for e in recorder.of_type("input_audio.append")[1:]
    )
    assert 0 < len(resampled) <= 1.5 * 6400
    assert len(resampled) % 2 == 0


# ---------------------------------------------------------------------------
# Projected turns → frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assistant_turn_brackets_text_with_one_response_and_tts_pair():
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    await _drive(
        service,
        [
            _turn_created("turn_a", "assistant", "Hi"),
            _turn_delta("turn_a", " there"),
            _turn_done("turn_a", "assistant", "Hi there"),
        ],
    )

    frames = [f for f, _ in recorder.frames]
    assert [type(f) for f in frames] == [
        LLMFullResponseStartFrame,
        TTSStartedFrame,
        LLMTextFrame,
        TTSTextFrame,
        LLMTextFrame,
        TTSTextFrame,
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]
    llm_text = recorder.of_types(LLMTextFrame)
    assert [f.text for f in llm_text] == ["Hi", " there"]
    assert all(f.append_to_context is False for f in llm_text)
    tts_text = recorder.of_types(TTSTextFrame)
    assert [f.text for f in tts_text] == ["Hi", " there"]
    assert all(f.includes_inter_frame_spaces for f in tts_text)


@pytest.mark.asyncio
async def test_user_turn_emits_proposed_speaking_frames_and_transcriptions():
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    await _drive(
        service,
        [
            _turn_created("turn_u", "user", "what's"),
            _turn_delta("turn_u", " the weather"),
            _turn_done("turn_u", "user", "what's the weather"),
        ],
    )

    started = recorder.of_types(ProposedUserStartedSpeakingFrame)
    stopped = recorder.of_types(ProposedUserStoppedSpeakingFrame)
    assert len(started) == 2 and len(stopped) == 2  # broadcast: one copy each direction

    interim = [(f, d) for f, d in recorder.frames if isinstance(f, InterimTranscriptionFrame)]
    assert [f.text for f, _ in interim] == ["what's", "what's the weather"]
    assert all(d == FrameDirection.UPSTREAM for _, d in interim)

    ((final, direction),) = [
        (f, d) for f, d in recorder.frames if isinstance(f, TranscriptionFrame)
    ]
    assert final.text == "what's the weather"
    assert direction == FrameDirection.UPSTREAM

    # Transcription precedes the stop proposal.
    types = [type(f) for f, _ in recorder.frames]
    assert types.index(TranscriptionFrame) < types.index(ProposedUserStoppedSpeakingFrame)


def test_metadata_frame_recommends_external_turns_without_interruptions():
    service = _make_service()
    frame = service.service_metadata_frame()
    assert frame.is_realtime_service is True
    assert frame.user_turn_strategies is not None
    assert frame.user_turn_strategies.enable_interruptions is False


# ---------------------------------------------------------------------------
# Responses delegation: function calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_function_call_item_runs_the_function():
    service = _make_service(delegation=_responses_delegation())
    service._context = LLMContext([], tools=[_weather_tool()])
    service.run_function_calls = AsyncMock()

    await _drive(
        service,
        [
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"location": "Seattle"}',
                },
            },
        ],
    )

    service.run_function_calls.assert_awaited_once()
    (calls,) = service.run_function_calls.call_args.args
    (call,) = calls
    assert call.tool_call_id == "call_1"
    assert call.function_name == "get_weather"
    assert call.arguments == {"location": "Seattle"}
    assert call.context is service._context
    assert "call_1" in service._open_function_calls


@pytest.mark.asyncio
async def test_function_call_result_is_sent_as_output_immediately():
    service = _make_service(delegation=_responses_delegation())
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service._open_function_calls.add("call_1")

    frame = FunctionCallResultFrame(
        function_name="get_weather",
        tool_call_id="call_1",
        arguments={"location": "Seattle"},
        result={"temp": 62},
    )
    await service.push_frame(frame, FrameDirection.UPSTREAM)
    assert recorder.events == []  # only the downstream broadcast copy is answered
    await service.push_frame(frame)

    (output,) = recorder.of_type("delegation.function_call_output.create")
    assert output["item"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"temp": 62}',
    }
    assert "call_1" not in service._open_function_calls


@pytest.mark.asyncio
async def test_intermediate_function_call_result_is_dropped():
    service = _make_service(delegation=_responses_delegation())
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service._open_function_calls.add("call_1")

    await service.push_frame(
        FunctionCallResultFrame(
            function_name="get_weather",
            tool_call_id="call_1",
            arguments={},
            result={"progress": "50%"},
            properties=FunctionCallResultProperties(is_final=False),
        )
    )

    assert recorder.events == []
    assert "call_1" in service._open_function_calls


@pytest.mark.asyncio
async def test_cancelled_function_call_reports_cancellation_to_the_backend():
    service = _make_service(delegation=_responses_delegation())
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service._open_function_calls.add("call_1")

    await service.push_frame(
        FunctionCallCancelFrame(function_name="get_weather", tool_call_id="call_1")
    )

    (output,) = recorder.of_type("delegation.function_call_output.create")
    assert output["item"]["call_id"] == "call_1"
    assert "cancelled" in json.loads(output["item"]["output"])["error"]


@pytest.mark.asyncio
async def test_results_for_unknown_calls_are_ignored():
    service = _make_service(delegation=_responses_delegation())
    recorder = _EventRecorder()
    service.send_client_event = recorder

    await service.push_frame(
        FunctionCallResultFrame(
            function_name="other", tool_call_id="call_x", arguments={}, result="ok"
        )
    )

    assert recorder.events == []


# ---------------------------------------------------------------------------
# Client delegation without a backend, errors, usage, close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_delegation_without_backend_is_declined():
    service = _make_service()
    recorder = _EventRecorder()
    service.send_client_event = recorder

    await _drive(
        service,
        [
            {
                "type": "delegation.created",
                "offset_ms": 1000,
                "item": {
                    "id": "item_d1",
                    "type": "delegation",
                    "target": "client",
                    "content": [{"type": "input_text", "text": "What is the weather?"}],
                },
            }
        ],
    )

    (append,) = recorder.of_type("delegation.context.append")
    assert append["delegation_item_id"] == "item_d1"
    assert append["channel"] == "commentary"


@pytest.mark.asyncio
async def test_error_before_session_start_is_permanent_and_after_is_not():
    service = _make_service()
    service.push_error = AsyncMock()
    error = {
        "type": "error",
        "error": {"type": "invalid_request_error", "code": "bad", "message": "nope"},
    }

    await _drive(service, [error])
    assert service.push_error.call_args.kwargs["force_treat_as_permanent"] is True

    await _drive(service, [_session_started(), error])
    assert service._session_started is True
    assert service.push_error.call_args.kwargs.get("force_treat_as_permanent", False) is False


@pytest.mark.asyncio
async def test_unknown_and_response_events_do_not_break_the_receive_loop():
    service = _make_service()
    service.push_frame = _FrameRecorder()

    await _drive(
        service,
        [
            {"type": "some.future.event", "payload": 1},
            {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
            "not json",
            _session_started(),
        ],
    )

    assert service._session_started is True


def test_parse_server_event_routes_response_events_and_unknown_types():
    response = events.parse_server_event(json.dumps({"type": "response.completed", "response": {}}))
    assert isinstance(response, events.ResponseEvent)
    unknown = events.parse_server_event(json.dumps({"type": "brand.new", "x": 1}))
    assert isinstance(unknown, events.UnknownServerEvent)
    with pytest.raises(ValueError):
        events.parse_server_event("[]")


@pytest.mark.asyncio
async def test_usage_reports_deltas_between_cumulative_updates():
    service = _make_service()
    service.start_llm_usage_metrics = AsyncMock()

    def usage(total, inp, out, in_audio, out_audio, cached):
        return {
            "type": "session.usage.updated",
            "usage": {
                "total_tokens": total,
                "input_tokens": inp,
                "output_tokens": out,
                "input_token_details": {"audio_tokens": in_audio, "cached_tokens": cached},
                "output_token_details": {"audio_tokens": out_audio},
            },
        }

    await _drive(service, [usage(100, 60, 40, 50, 30, 10), usage(150, 90, 60, 70, 45, 25)])

    first, second = [c.args[0] for c in service.start_llm_usage_metrics.call_args_list]
    assert (first.total_tokens, first.prompt_tokens, first.completion_tokens) == (100, 60, 40)
    assert (first.input_audio_tokens, first.output_audio_tokens) == (50, 30)
    assert first.cache_read_input_tokens == 10
    assert (second.total_tokens, second.prompt_tokens, second.completion_tokens) == (50, 30, 20)
    assert (second.input_audio_tokens, second.output_audio_tokens) == (20, 15)
    assert second.cache_read_input_tokens == 15


@pytest.mark.asyncio
async def test_close_session_waits_for_session_closed(monkeypatch):
    service = _make_service()
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service._websocket = object()
    service._session_started = True

    async def close_from_server():
        await asyncio.sleep(0)
        await service._handle_server_event(
            events.parse_server_event(
                json.dumps({"type": "session.closed", "reason": "client_request"})
            )
        )

    await asyncio.gather(service._close_session(), close_from_server())

    assert recorder.of_type("session.close")
    assert service._session_started is False

    # Without a session.closed the wait times out and returns.
    service._session_started = True
    monkeypatch.setattr(live_llm, "SESSION_CLOSE_TIMEOUT_SECS", 0.01)
    await service._close_session()
    assert len(recorder.of_type("session.close")) == 2


@pytest.mark.asyncio
async def test_reset_conversation_starts_a_new_session_from_the_current_context():
    service = _make_service()
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service._close_session = AsyncMock()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    context = LLMContext([{"role": "user", "content": "hi"}])
    await service._handle_context(context)
    service._session_started = True

    context.set_messages([{"role": "assistant", "content": "Restored history."}])
    await service.reset_conversation()

    service._close_session.assert_awaited_once()
    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()
    updates = recorder.of_type("session.update")
    assert len(updates) == 2
    assert updates[1]["session"]["initial_items"] == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Restored history."}],
        }
    ]
    assert service._needs_session_config is False
