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
    LLMSetToolsFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    SpeechOutputAudioRawFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.job_context import JobError
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.live import events
from pipecat.services.openai.live import llm as live_llm
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.base_object import BaseObject
from pipecat.workers.llm import BackendOutput

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


#: Turn gap short enough to keep tests quick, long enough to survive scheduling.
TEST_TURN_GAP_SECS = 0.05


async def _make_service_with_tasks(*, delegation=None, settings=None) -> OpenAILiveLLMService:
    """A service wired to a task manager, for the paths that run turn timers."""
    settings = settings or OpenAILiveLLMService.Settings()
    settings.transcript_turn_gap_secs = TEST_TURN_GAP_SECS
    service = _make_service(delegation=delegation, settings=settings)
    await BaseObject.setup(service, TaskManager())
    return service


async def _let_turns_close(service: OpenAILiveLLMService) -> None:
    """Wait out the turn gap, so any open turn is closed by its timer."""
    await asyncio.sleep(TEST_TURN_GAP_SECS * 3)


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

    async def close(self) -> None:
        self._messages.clear()


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
        self.events.append(event.to_payload())

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]


async def _drive(service: OpenAILiveLLMService, scripted: list[dict[str, Any]]) -> None:
    """Feed scripted server-event dicts through the receive handler."""
    service._websocket = _FakeWebSocket([json.dumps(e) for e in scripted])
    await service._receive_task_handler()


def _transcript_delta(role: str, delta: str, *, start_ms: int = 0) -> dict[str, Any]:
    kind = "input" if role == "user" else "output"
    return {
        "type": f"session.{kind}_transcript.delta",
        "delta": delta,
        "start_ms": start_ms,
        "end_ms": start_ms + 200,
    }


def _delegation_created(delegation_id: str, target: str = "client") -> dict[str, Any]:
    delegation: dict[str, Any] = {"id": delegation_id, "type": "delegation", "target": target}
    if target == "responses":
        delegation["response_id"] = "resp_1"
    return {"type": "session.delegation.created", "offset_ms": 1000, "delegation": delegation}


def _response_event(inner: dict[str, Any], delegation_id: str = "item_d1") -> dict[str, Any]:
    return {"type": "response.event", "delegation_id": delegation_id, "event": inner}


def _session_started() -> dict[str, Any]:
    return {
        "type": "session.started",
        "session": {"id": "live_123", "model": "gpt-live-1-diamond-alpha"},
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

    (start,) = recorder.of_type("session.start")
    session = start["session"]
    assert session["model"] == live_llm.DEFAULT_MODEL
    assert session["instructions"] == "Be brief."
    assert session["audio"] == {"output": {"voice": "cedar"}}
    assert session["delegation"] == {"type": "client"}
    assert session["input"] == [
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
async def test_no_voice_setting_leaves_the_choice_to_the_api():
    """Without a voice, no audio config is sent and the API picks its default."""
    service = _make_service()
    recorder = _EventRecorder()
    service.send_client_event = recorder

    await service._handle_context(LLMContext([{"role": "system", "content": "Be brief."}]))

    (start,) = recorder.of_type("session.start")
    assert "audio" not in start["session"]


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

    (start,) = recorder.of_type("session.start")
    session = start["session"]
    assert "input" not in session
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

    (start,) = recorder.of_type("session.start")
    assert start["session"]["instructions"] == "From settings."


@pytest.mark.asyncio
async def test_startup_history_keeps_the_most_recent_128():
    service = _make_service()
    recorder = _EventRecorder()
    service.send_client_event = recorder

    messages = [{"role": "user", "content": f"message {i}"} for i in range(200)]
    await service._handle_context(LLMContext(messages))

    (start,) = recorder.of_type("session.start")
    items = start["session"]["input"]
    assert len(items) == events.MAX_INPUT_ITEMS
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
    assert recorder.of_type("session.update") == []

    # New tool set: sparse delegation update carrying only the tools.
    context.set_tools(ToolsSchema(standard_tools=[]))
    await service._handle_context(context)
    (update,) = recorder.of_type("session.update")
    assert update["session"] == {"delegation": {"type": "responses", "responses": {"tools": []}}}


@pytest.mark.asyncio
async def test_set_tools_frame_updates_the_session_without_a_context_frame():
    """A continuous session gets no context frame per turn, so the frame is what prompts the update."""
    service = await _make_service_with_tasks(delegation=_responses_delegation())
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service.push_frame = _FrameRecorder()

    context = LLMContext([{"role": "user", "content": "hi"}], tools=[_weather_tool()])
    await service._handle_context(context)
    service._session_started = True

    # The user aggregator sets the new tools on the shared context and forwards
    # the frame; the service turns that into a session update.
    new_tools = ToolsSchema(standard_tools=[])
    context.set_tools(new_tools)
    await service.process_frame(LLMSetToolsFrame(tools=new_tools), FrameDirection.DOWNSTREAM)

    (update,) = recorder.of_type("session.update")
    assert update["session"] == {"delegation": {"type": "responses", "responses": {"tools": []}}}


@pytest.mark.asyncio
async def test_a_failed_disconnect_still_allows_the_next_session_to_send():
    """`_disconnecting` gates every send, so it is cleared even when teardown fails."""
    service = await _make_service_with_tasks()
    service.push_error = AsyncMock()

    class _FailingWebSocket:
        async def close(self):
            raise RuntimeError("close failed")

    service._websocket = _FailingWebSocket()
    await service._disconnect()

    service.push_error.assert_awaited_once()
    assert service._disconnecting is False


def _error(message: str = "boom") -> dict[str, Any]:
    return {"type": "error", "error": {"type": "invalid_request", "message": message}}


@pytest.mark.asyncio
async def test_a_startup_error_is_fatal_for_every_session_not_just_the_first():
    """A reset asks for a new session, so its startup can fail in turn."""
    service = await _make_service_with_tasks()
    service.send_client_event = _EventRecorder()
    service.push_frame = _FrameRecorder()
    service.push_error = AsyncMock()
    service._connect = AsyncMock()

    context = LLMContext([{"role": "user", "content": "hi"}])
    await service._handle_context(context)

    # First session: an error before it starts is a startup failure.
    await _drive(service, [_error()])
    assert service.push_error.await_args.kwargs.get("force_treat_as_permanent") is True

    # It starts, and a later error is just an error.
    await _drive(service, [_session_started(), _error()])
    assert service.push_error.await_args.kwargs.get("force_treat_as_permanent") is not True

    # An error while it shuts down is not a startup failure either.
    await _drive(service, [{"type": "session.closed", "reason": "done"}, _error()])
    assert service.push_error.await_args.kwargs.get("force_treat_as_permanent") is not True

    # A reset asks for a second session; its startup can fail the same way.
    await service.reset_conversation()
    await _drive(service, [_error()])
    assert service.push_error.await_args.kwargs.get("force_treat_as_permanent") is True


@pytest.mark.asyncio
async def test_the_opening_nudge_carries_a_null_delegation_id():
    """The append is session-wide, and the API requires the field even so."""
    service = await _make_service_with_tasks()
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service.push_frame = _FrameRecorder()

    context = LLMContext([{"role": "developer", "content": "Greet the user."}])
    await service._handle_context(context)
    await _drive(service, [_session_started()])

    (append,) = recorder.of_type("session.commentary.append")
    assert append["delegation_id"] is None
    assert append["content"] == "Greet the user."


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_audio_delta_pushes_24khz_speech_audio_frame():
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder
    audio = b"\x01\x02" * 240

    await _drive(
        service,
        [
            {
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(audio).decode("ascii"),
            }
        ],
    )

    (frame,) = recorder.of_types(SpeechOutputAudioRawFrame)
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
    (append,) = recorder.of_type("session.input_audio.append")
    assert base64.b64decode(append["audio"]) == audio_24k

    # 16 kHz input is resampled to 24 kHz. The stream resampler emits in its
    # own chunk sizes with some latency, so feed 200 ms and check the total.
    for _ in range(10):
        await service._send_user_audio(InputAudioRawFrame(b"\x00\x01" * 320, 16000, 1))
    resampled = b"".join(
        base64.b64decode(e["audio"]) for e in recorder.of_type("session.input_audio.append")[1:]
    )
    assert 0 < len(resampled) <= 1.5 * 6400
    assert len(resampled) % 2 == 0


# ---------------------------------------------------------------------------
# Transcript fragments → turn frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assistant_turn_brackets_text_with_one_response_and_tts_pair():
    service = await _make_service_with_tasks()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    await _drive(
        service,
        [
            _transcript_delta("assistant", "Hi"),
            _transcript_delta("assistant", " there", start_ms=200),
        ],
    )
    await _let_turns_close(service)

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
    service = await _make_service_with_tasks()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    await _drive(
        service,
        [
            _transcript_delta("user", "what's"),
            _transcript_delta("user", " the weather", start_ms=200),
        ],
    )
    await _let_turns_close(service)

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


@pytest.mark.asyncio
async def test_a_gap_ends_a_turn_and_the_next_fragment_starts_another():
    service = await _make_service_with_tasks()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    await _drive(service, [_transcript_delta("assistant", "First.")])
    await _let_turns_close(service)
    await _drive(service, [_transcript_delta("assistant", "Second.", start_ms=5000)])
    await _let_turns_close(service)

    assert len(recorder.of_types(LLMFullResponseStartFrame)) == 2
    assert len(recorder.of_types(LLMFullResponseEndFrame)) == 2
    assert [f.text for f in recorder.of_types(TTSTextFrame)] == ["First.", "Second."]


@pytest.mark.asyncio
async def test_both_speakers_can_hold_a_turn_at_once():
    """Full duplex: the two directions are grouped independently."""
    service = await _make_service_with_tasks()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    await _drive(
        service,
        [
            _transcript_delta("assistant", "Let me check"),
            _transcript_delta("user", "actually wait"),
        ],
    )
    await _let_turns_close(service)

    assert [f.text for f in recorder.of_types(TTSTextFrame)] == ["Let me check"]
    ((final, _),) = [(f, d) for f, d in recorder.frames if isinstance(f, TranscriptionFrame)]
    assert final.text == "actually wait"


def test_metadata_frame_recommends_external_turns_without_interruptions():
    service = _make_service()
    frame = service.service_metadata_frame()
    assert frame.is_realtime_service is False
    assert frame.user_turn_strategies is not None
    assert frame.user_turn_strategies.enable_interruptions is False


# ---------------------------------------------------------------------------
# Responses delegation: function calls
# ---------------------------------------------------------------------------


def _function_call_item(call_id: str, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": f"fc_{call_id}",
            "type": "function_call",
            "status": status,
            "call_id": call_id,
            "name": "get_weather",
            "arguments": '{"location": "Seattle"}',
        },
    }


def _response_created(response_id: str = "resp_1") -> dict[str, Any]:
    return {"type": "response.created", "response": {"id": response_id}}


def _response_completed(response_id: str = "resp_1", **response) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {"id": response_id, "output": [], **response},
    }


@pytest.mark.asyncio
async def test_completed_function_call_item_runs_the_function():
    service = _make_service(delegation=_responses_delegation())
    service._context = LLMContext([], tools=[_weather_tool()])
    service.run_function_calls = AsyncMock()

    await _drive(
        service,
        [
            _response_event(_response_created()),
            _response_event(_function_call_item("call_1", status="in_progress")),
            _response_event(_function_call_item("call_1")),
        ],
    )

    service.run_function_calls.assert_awaited_once()
    (calls,) = service.run_function_calls.call_args.args
    (call,) = calls
    assert call.tool_call_id == "call_1"
    assert call.function_name == "get_weather"
    assert call.arguments == {"location": "Seattle"}
    assert call.context is service._context
    assert service._open_function_calls["call_1"] == "item_d1"


@pytest.mark.asyncio
async def test_function_result_is_queued_and_the_response_continued_once_all_are_in():
    service = _make_service(delegation=_responses_delegation())
    service._context = LLMContext([], tools=[_weather_tool()])
    service.run_function_calls = AsyncMock()
    recorder = _EventRecorder()
    service.send_client_event = recorder

    await _drive(
        service,
        [
            _response_event(_response_created()),
            _response_event(_function_call_item("call_1")),
            _response_event(_function_call_item("call_2")),
            _response_event(_response_completed()),
        ],
    )

    frame = FunctionCallResultFrame(
        function_name="get_weather",
        tool_call_id="call_1",
        arguments={"location": "Seattle"},
        result={"temp": 62},
    )
    await service.push_frame(frame, FrameDirection.UPSTREAM)
    assert recorder.events == []  # only the downstream broadcast copy is answered

    await service.push_frame(frame)
    (item,) = recorder.of_type("response.item.create")
    assert item["item"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"temp": 62}',
    }
    # The other call is still outstanding, so the response is not continued yet.
    assert recorder.of_type("response.create") == []

    await service.push_frame(
        FunctionCallResultFrame(
            function_name="get_weather",
            tool_call_id="call_2",
            arguments={"location": "Boston"},
            result={"temp": 51},
        )
    )
    assert len(recorder.of_type("response.item.create")) == 2
    assert len(recorder.of_type("response.create")) == 1


@pytest.mark.asyncio
async def test_a_response_that_asked_for_nothing_is_not_continued():
    """An empty terminal output list is not a reason to continue a response."""
    service = _make_service(delegation=_responses_delegation())
    recorder = _EventRecorder()
    service.send_client_event = recorder

    await _drive(
        service,
        [_response_event(_response_created()), _response_event(_response_completed())],
    )

    assert recorder.of_type("response.create") == []


@pytest.mark.asyncio
async def test_backend_token_usage_is_reported_from_the_completed_response():
    service = _make_service(delegation=_responses_delegation())
    service.start_llm_usage_metrics = AsyncMock()

    await _drive(
        service,
        [
            _response_event(_response_created()),
            _response_event(
                _response_completed(
                    usage={
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "total_tokens": 150,
                        "input_tokens_details": {"cached_tokens": 100},
                        "output_tokens_details": {"reasoning_tokens": 12},
                    }
                )
            ),
        ],
    )

    (tokens,) = service.start_llm_usage_metrics.call_args.args
    assert tokens.prompt_tokens == 120
    assert tokens.completion_tokens == 30
    assert tokens.total_tokens == 150
    assert tokens.cache_read_input_tokens == 100
    assert tokens.reasoning_tokens == 12


@pytest.mark.asyncio
async def test_intermediate_function_call_result_is_dropped():
    service = _make_service(delegation=_responses_delegation())
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service._open_function_calls["call_1"] = "item_d1"

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
    service._open_function_calls["call_1"] = "item_d1"

    await service.push_frame(
        FunctionCallCancelFrame(function_name="get_weather", tool_call_id="call_1")
    )

    (item,) = recorder.of_type("response.item.create")
    assert item["item"]["call_id"] == "call_1"
    assert "cancelled" in json.loads(item["item"]["output"])["error"]


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
            _delegation_created("item_d1"),
        ],
    )

    (append,) = recorder.of_type("session.commentary.append")
    assert append["delegation_id"] == "item_d1"
    assert "No backend" in append["content"]


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
            _response_event({"type": "response.output_text.delta", "delta": "hi"}),
            "not json",
            _session_started(),
        ],
    )

    assert service._session_started is True


def test_parse_server_event_unwraps_response_envelopes_and_tolerates_unknown_types():
    envelope = events.parse_server_event(
        json.dumps(_response_event({"type": "response.completed", "response": {"id": "resp_1"}}))
    )
    assert isinstance(envelope, events.ResponseEventEnvelope)
    assert envelope.delegation_id == "item_d1"
    assert envelope.inner_type == "response.completed"

    unknown = events.parse_server_event(json.dumps({"type": "brand.new", "x": 1}))
    assert isinstance(unknown, events.UnknownServerEvent)
    with pytest.raises(ValueError):
        events.parse_server_event("[]")


def test_errors_correlate_to_the_command_that_caused_them():
    error = events.parse_server_event(
        json.dumps(
            {
                "type": "error",
                "event_id": "event_error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "immutable_field_update",
                    "message": "nope",
                    "client_event_id": "event_update",
                },
            }
        )
    )
    assert isinstance(error, events.ErrorEvent)
    assert error.error.client_event_id == "event_update"


@pytest.mark.asyncio
async def test_live_usage_is_reported_as_cumulative_seconds():
    """The live model bills duration, not tokens; backend tokens arrive separately."""
    service = _make_service()
    service.start_llm_usage_metrics = AsyncMock()

    await _drive(
        service,
        [
            {"type": "session.usage.updated", "usage": {"seconds": 12.4}},
            {
                "type": "session.usage.updated",
                "usage": {"seconds": 30.1},
                "context_window": {"usage_ratio": 0.42},
            },
        ],
    )

    service.start_llm_usage_metrics.assert_not_called()


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
    service = await _make_service_with_tasks()
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service._close_session = AsyncMock()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    context = LLMContext([{"role": "user", "content": "hi"}])
    await service._handle_context(context)
    service._session_started = True

    context.set_messages([{"role": "assistant", "content": "Restored history."}])
    frames = _FrameRecorder()
    service.push_frame = frames
    service._assistant_turn.open = True
    await service.reset_conversation()

    # The old session is dropped, not drained, and its open assistant turn is closed.
    service._close_session.assert_not_awaited()
    service._disconnect.assert_awaited_once()
    assert [type(f) for f, _ in frames.frames] == [TTSStoppedFrame, LLMFullResponseEndFrame]
    service._connect.assert_awaited_once()
    starts = recorder.of_type("session.start")
    assert len(starts) == 2
    assert starts[1]["session"]["input"] == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Restored history."}],
        }
    ]
    assert service._needs_session_config is False


# ---------------------------------------------------------------------------
# Client delegation
# ---------------------------------------------------------------------------


class _FakeBackend:
    name = "backend"


async def _client_delegation_service(monkeypatch, _delegate_to_backend):
    service = await _make_service_with_tasks(
        delegation=OpenAILiveLLMService.ClientDelegation(backend=_FakeBackend(), timeout_secs=5)
    )
    recorder = _EventRecorder()
    service.send_client_event = recorder
    monkeypatch.setattr(live_llm, "_delegate_to_backend", _delegate_to_backend)
    monkeypatch.setattr(type(service), "pipeline_worker", property(lambda self: "worker"))
    return service, recorder


def _client_delegation(delegation_id: str) -> events.DelegationMetadata:
    return events.DelegationMetadata(id=delegation_id, target="client")


@pytest.mark.asyncio
async def test_client_delegation_sends_the_fragments_since_the_last_one(monkeypatch):
    calls = []

    async def fake_delegate_to_backend(worker, backend_name, *, request, on_update, timeout_secs):
        calls.append((worker, backend_name, request, timeout_secs))
        await on_update(BackendOutput(text="Checking the weather.", prefers_spoken=True))
        await on_update(BackendOutput(text="Still looking.", is_thought=True, prefers_spoken=False))
        await on_update(
            BackendOutput(
                text="It's 62 and raining in Seattle.", is_final=True, prefers_spoken=True
            )
        )
        return "It's 62 and raining in Seattle."

    service, recorder = await _client_delegation_service(monkeypatch, fake_delegate_to_backend)

    await _drive(
        service,
        [
            _transcript_delta("user", "what's the weather in seattle"),
            _transcript_delta("assistant", "Let me check.", start_ms=200),
        ],
    )
    await service._run_client_delegation(_client_delegation("item_d1"))

    # No task text: the delegation names none, so the backend is handed the
    # conversation rendered as a transcript and works the request out from it.
    assert calls == [
        (
            "worker",
            "backend",
            "Voice conversation so far:\n"
            "USER: what's the weather in seattle\n"
            "ASSISTANT: Let me check.\n"
            "\n"
            "Act on the user's most recent request in the conversation above.",
            5,
        )
    ]
    assert service._transcript_fragments == []
    thinking = [
        (e["delegation_id"], e["content"]) for e in recorder.of_type("session.thinking.append")
    ]
    commentary = [
        (e["delegation_id"], e["content"]) for e in recorder.of_type("session.commentary.append")
    ]
    assert thinking == [("item_d1", "Still looking.")]
    assert commentary == [
        ("item_d1", "Checking the weather."),
        ("item_d1", "It's 62 and raining in Seattle."),
    ]


@pytest.mark.asyncio
async def test_the_backend_reads_whole_utterances_not_fragments(monkeypatch):
    """Frame-boundary fragments are joined back up, spacing and all."""
    calls = []

    async def fake_delegate_to_backend(worker, backend_name, *, request, on_update, timeout_secs):
        calls.append(request)
        return ""

    service, _ = await _client_delegation_service(monkeypatch, fake_delegate_to_backend)

    await _drive(
        service,
        [
            _transcript_delta("assistant", "Hey there"),
            _transcript_delta("assistant", "!", start_ms=200),
            _transcript_delta("user", "Get", start_ms=400),
            _transcript_delta("user", " me the", start_ms=600),
            _transcript_delta("user", " weather in", start_ms=800),
            _transcript_delta("user", " Washington", start_ms=1000),
            _transcript_delta("user", ", DC", start_ms=1200),
        ],
    )
    await service._run_client_delegation(_client_delegation("item_d1"))

    assert calls == [
        "Voice conversation so far:\n"
        "ASSISTANT: Hey there!\n"
        "USER: Get me the weather in Washington, DC\n"
        "\n"
        "Act on the user's most recent request in the conversation above."
    ]


@pytest.mark.asyncio
async def test_a_reset_starts_the_next_delegation_transcript_afresh(monkeypatch):
    """A new session has no previous delegation for a transcript to run from."""
    requests: list[str] = []

    async def fake_delegate_to_backend(worker, backend_name, *, request, on_update, timeout_secs):
        requests.append(request)
        return ""

    service, _ = await _client_delegation_service(monkeypatch, fake_delegate_to_backend)
    service._connect = AsyncMock()

    await _drive(service, [_transcript_delta("user", "what's the weather in seattle")])
    await service._run_client_delegation(_client_delegation("item_d1"))

    await _drive(service, [_transcript_delta("user", "and in boston", start_ms=2000)])
    await service._run_client_delegation(_client_delegation("item_d2"))

    await service.reset_conversation()

    await _drive(service, [_transcript_delta("user", "let's start over", start_ms=4000)])
    await service._run_client_delegation(_client_delegation("item_d3"))

    assert requests[0].startswith("Voice conversation so far:")
    assert requests[1].startswith("Voice conversation since the previous delegation:")
    assert requests[2].startswith("Voice conversation so far:")


@pytest.mark.asyncio
async def test_client_delegation_failure_is_reported_to_the_model(monkeypatch):
    async def failing_delegate_to_backend(*args, **kwargs):
        raise JobError("timed out")

    service, recorder = await _client_delegation_service(monkeypatch, failing_delegate_to_backend)
    service.push_error = AsyncMock()

    await service._run_client_delegation(_client_delegation("item_d1"))

    # The model is told the work failed; the detail goes to the error instead.
    (append,) = recorder.of_type("session.commentary.append")
    assert "timed out" not in append["content"]
    service.push_error.assert_awaited_once()
    assert "timed out" in service.push_error.await_args.kwargs["error_msg"]


@pytest.mark.asyncio
async def test_long_delegation_results_are_chunked_at_sentence_boundaries(monkeypatch):
    async def _delegate_to_backend(*args, on_update, **kwargs):
        text = " ".join(f"Sentence number {i} is here." for i in range(120))
        await on_update(BackendOutput(text=text, prefers_spoken=True))
        return ""

    service, recorder = await _client_delegation_service(monkeypatch, _delegate_to_backend)
    await service._run_client_delegation(_client_delegation("item_d1"))

    appends = recorder.of_type("session.commentary.append")
    assert len(appends) > 1
    for append in appends:
        assert len(append["content"]) <= live_llm.MAX_CONTEXT_APPEND_CHARS
        assert append["content"].endswith(".")
    assert " ".join(a["content"] for a in appends).count("Sentence number") == 120


def test_chunk_text_splits_overlong_sentences_on_whitespace():
    words = " ".join(["word"] * 400)
    chunks = live_llm._chunk_text(words, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks) == words
    assert live_llm._chunk_text("   ", 100) == []
