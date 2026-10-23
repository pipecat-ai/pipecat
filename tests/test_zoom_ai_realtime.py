#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ZoomAIRealtimeLLMService and its session models.

The service is driven without a network: session updates are captured from
``_ws_send``, and server events are fed through the connection adapter that
sits between the Voice Agent and the OpenAI Realtime event handling.
"""

import base64
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.openai.realtime import events as openai_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.zoom_ai.realtime import events
from pipecat.services.zoom_ai.realtime import llm as llm_module
from pipecat.services.zoom_ai.realtime.llm import (
    ZoomAIRealtimeLLMService,
    _ZoomAIRealtimeWebSocket,
)
from pipecat.utils.errors import ErrorCategory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ErrorRecorder:
    """Records pushed errors; ``messages`` leaves out the session's tracking id."""

    def __init__(self):
        self.messages: list[str] = []
        self.raw: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    async def __call__(self, error_msg: str, **kwargs):
        self.raw.append(error_msg)
        self.messages.append(re.sub(r" \(tracking_id=[^)]*\)$", "", error_msg))
        self.kwargs.append(kwargs)


def _service(**settings) -> ZoomAIRealtimeLLMService:
    return ZoomAIRealtimeLLMService(
        api_key="test-key",
        settings=ZoomAIRealtimeLLMService.Settings(**settings) if settings else None,
    )


async def _session_updates(service: ZoomAIRealtimeLLMService) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def _ws_send(message):
        sent.append(message)

    service._ws_send = _ws_send  # type: ignore[method-assign]
    await service._send_session_update()
    return [m["session"] for m in sent if m["type"] == "session.update"]


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any]]):
        self._messages = [json.dumps(m) for m in messages]
        self.sent: list[str] = []

    async def send(self, message: str):
        self.sent.append(message)

    async def close(self):
        pass

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m

        return _gen()


async def _adapt(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []

    async def _on_error(error: dict[str, Any], event_ids: set[str]):
        errors.append(error["message"])

    ws = _ZoomAIRealtimeWebSocket(_FakeWebSocket(messages), on_error=_on_error)
    return [json.loads(m) async for m in ws], errors


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------


def test_unset_fields_are_omitted_and_type_is_always_sent():
    update = events.SessionUpdateEvent(
        session=events.SessionProperties(
            listen=events.Listen(transcription=events.Transcription(language="en")),
            speak=events.Speak(voice="jenna"),
        )
    )
    assert update.to_wire() == {
        "type": "session.update",
        "session": {
            "listen": {"transcription": {"language": "en"}},
            "speak": {"voice": "jenna"},
        },
    }


def test_speak_set_to_none_is_sent_as_null():
    session = events.SessionProperties(speak=None)
    assert events.SessionUpdateEvent(session=session).to_wire()["session"] == {"speak": None}


def test_provider_carries_headers_and_metadata():
    provider = events.Provider(
        type="openai",
        model="gpt-5",
        headers={"X-Api-Key": "k"},
        metadata={"context_window": 1000000},
    )
    assert provider.model_dump(exclude_unset=True) == {
        "type": "openai",
        "model": "gpt-5",
        "headers": {"X-Api-Key": "k"},
        "metadata": {"context_window": 1000000},
    }


# ---------------------------------------------------------------------------
# Service configuration and session updates
# ---------------------------------------------------------------------------


def test_defaults():
    service = _service()
    # No ?model= query: the model is in the session.
    assert service.base_url == "wss://api.zoom.us/v2/ws/aiservices/voice-agent/live"
    assert service._settings.model == "gemma-4"


@pytest.mark.asyncio
async def test_session_update_carries_listen_think_and_speak(monkeypatch):
    async def _parent_setup(self, setup):
        pass

    monkeypatch.setattr(OpenAIRealtimeLLMService, "setup", _parent_setup)
    service = _service(
        system_instruction="Be brief.",
        listen=events.Listen(
            transcription=events.Transcription(
                language="en", vocabulary=events.Vocabulary(phrases=["Pipecat"])
            )
        ),
        speak=events.Speak(voice="ariel"),
    )
    await service.setup(SimpleNamespace(audio_in_sample_rate=8000))  # type: ignore[arg-type]
    (session,) = await _session_updates(service)
    assert session["listen"] == {
        "format": {"type": "pcm16", "rate": 8000},
        "transcription": {"language": "en", "vocabulary": {"phrases": ["Pipecat"]}},
    }
    assert session["think"]["provider"] == {"type": "zoom", "model": "gemma-4"}
    assert session["think"]["instructions"] == "Be brief."
    assert session["speak"] == {"format": {"type": "pcm16", "rate": 16000}, "voice": "ariel"}


@pytest.mark.asyncio
async def test_listen_is_left_out_once_the_session_is_configured():
    service = _service(listen=events.Listen(transcription=events.Transcription(language="en")))
    service._listen_configured = True
    (session,) = await _session_updates(service)
    assert "listen" not in session
    assert "think" in session
    assert "speak" not in session  # nothing about speaking changed


@pytest.mark.asyncio
async def test_voice_from_openai_audio_settings_is_used_when_speak_names_none():
    service = _service(
        session_properties=openai_events.SessionProperties(
            audio=openai_events.AudioConfiguration(
                output=openai_events.AudioOutput(voice="tony", speed=0.9)
            )
        )
    )
    (session,) = await _session_updates(service)
    assert session["speak"]["voice"] == "tony"
    assert "speed" not in session["speak"]  # not a Voice Agent setting


@pytest.mark.asyncio
async def test_text_only_sessions_send_speak_as_null():
    service = _service(
        session_properties=openai_events.SessionProperties(output_modalities=["text"])
    )
    (session,) = await _session_updates(service)
    assert session["speak"] is None


@pytest.mark.asyncio
async def test_think_carries_tool_choice_token_limit_and_reasoning():
    service = _service(
        session_properties=openai_events.SessionProperties(
            tool_choice="required",
            max_output_tokens=200,
            reasoning=openai_events.Reasoning(effort="low"),
        )
    )
    (session,) = await _session_updates(service)
    think = session["think"]
    assert think["tool_choice"] == "required"
    assert think["response_token_limit"] == 200
    assert think["reasoning"] == {"effort": "low"}


def test_turn_detection_is_always_the_servers():
    service = _service(
        session_properties=openai_events.SessionProperties(
            audio=openai_events.AudioConfiguration(
                input=openai_events.AudioInput(turn_detection=False)
            )
        )
    )
    assert not service._is_turn_detection_disabled()


@pytest.mark.asyncio
async def test_a_speak_update_sends_a_session_update():
    service = _service()
    sent: list[dict[str, Any]] = []

    async def _ws_send(message):
        sent.append(message)

    service._ws_send = _ws_send  # type: ignore[method-assign]
    await service._update_settings(
        ZoomAIRealtimeLLMService.Settings(speak=events.Speak(voice="alex"))
    )
    assert sent and sent[-1]["session"]["speak"]["voice"] == "alex"


@pytest.mark.asyncio
async def test_audio_waits_for_the_session_to_be_configured():
    service = _service()
    sent: list[Any] = []

    async def _send_client_event(event):
        sent.append(event)

    service.send_client_event = _send_client_event  # type: ignore[method-assign]
    frame = InputAudioRawFrame(audio=b"\x01\x00" * 160, sample_rate=16000, num_channels=1)
    await service._send_user_audio(frame)
    assert sent == []  # held until the session is ready
    assert bytes(service._pending_audio) == frame.audio
    service._api_session_ready = True
    await service._send_user_audio(frame)
    assert [type(e).__name__ for e in sent] == ["InputAudioBufferAppendEvent"]


@pytest.mark.asyncio
async def test_held_audio_is_sent_once_the_session_is_ready(monkeypatch):
    async def _parent_session_updated(self, evt):
        self._api_session_ready = True

    monkeypatch.setattr(
        OpenAIRealtimeLLMService, "_handle_evt_session_updated", _parent_session_updated
    )
    service = _service()
    sent: list[bytes] = []

    async def _send_client_event(event):
        sent.append(base64.b64decode(event.audio))

    service.send_client_event = _send_client_event  # type: ignore[method-assign]
    second = b"\x02\x00" * 16000  # one second at 16 kHz
    for n in range(5):  # five seconds while connecting; only the latest three are kept
        await service._send_user_audio(
            InputAudioRawFrame(audio=bytes([n, 0]) * 16000, sample_rate=16000, num_channels=1)
        )
    assert len(service._pending_audio) == 3 * len(second)
    await service._handle_evt_session_updated(SimpleNamespace())
    held = b"".join(sent)
    assert held == bytes([2, 0]) * 16000 + bytes([3, 0]) * 16000 + bytes([4, 0]) * 16000
    assert not service._pending_audio


# ---------------------------------------------------------------------------
# Server events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_item_created_is_delivered_as_item_added():
    out, _ = await _adapt(
        [
            {
                "type": "conversation.item.created",
                "event_id": "e1",
                "item": {"id": "i1", "type": "message", "role": "user", "content": []},
            }
        ]
    )
    assert out[0]["type"] == "conversation.item.added"
    openai_events.parse_server_event(json.dumps(out[0]))


@pytest.mark.asyncio
async def test_response_events_get_the_fields_the_event_models_require():
    out, _ = await _adapt(
        [
            {
                "type": "response.created",
                "response": {
                    "id": "r1",
                    "status": "in_progress",
                    # the Voice Agent's audio settings, not the OpenAI shape
                    "audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}},
                },
            },
            {
                "type": "response.done",
                "response": {
                    "id": "r1",
                    "status": "cancelled",
                    "usage": {"input_tokens": 3, "output_tokens": 0, "total_tokens": 3},
                },
            },
            {"type": "response.done", "response": {"id": "r2", "status": "failed"}},
        ]
    )
    for event in out:
        openai_events.parse_server_event(json.dumps(event))
    assert out[2]["response"]["status_details"]["error"]["message"]


@pytest.mark.asyncio
async def test_unknown_events_are_ignored():
    out, _ = await _adapt(
        [
            {"type": "input_audio_buffer.timeout_triggered"},
            {"type": "session.updated", "session": {}},
        ]
    )
    assert [e["type"] for e in out] == ["session.updated"]


@pytest.mark.asyncio
async def test_non_fatal_errors_are_reported_without_ending_the_session():
    out, errors = await _adapt(
        [
            {"type": "error", "error": {"type": "invalid_request_error", "message": "bad voice"}},
            {"type": "error", "error": {"code": "idle_timeout", "message": "idle", "fatal": True}},
        ]
    )
    assert errors == ["bad voice"]
    assert len(out) == 1 and out[0]["error"]["type"] == "server_error"
    openai_events.parse_server_event(json.dumps(out[0]))


@pytest.mark.asyncio
async def test_expected_errors_are_not_pushed_as_error_frames():
    service = _service()
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]
    for message in (
        "No active response to cancel",
        "Conversation already has an active response in progress",
        "speech synthesis failed",
    ):
        await service._handle_server_error({"type": "server_error", "message": message}, set())
    assert errors.messages == ["Zoom Voice Agent error: speech synthesis failed"]


@pytest.mark.asyncio
async def test_agent_audio_at_another_rate_is_resampled_to_24khz():
    pcm_16k = b"\x10\x00" * 1600  # 100 ms at 16 kHz
    out, _ = await _adapt(
        [
            {"type": "session.updated", "session": {"speak": {"format": {"rate": 16000}}}},
            *[
                {
                    "type": "response.output_audio.delta",
                    "response_id": "r1",
                    "item_id": "i1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": base64.b64encode(pcm_16k).decode(),
                }
            ]
            * 5,
        ]
    )
    resampled = sum(len(base64.b64decode(e["delta"])) for e in out[1:])
    assert resampled == pytest.approx(len(pcm_16k) * 5 * 1.5, rel=0.1)


@pytest.mark.asyncio
async def test_audio_deltas_that_end_mid_sample_are_joined():
    pcm = b"\x10\x00" * 8001  # 500 ms at 16 kHz, split below mid-sample

    def delta(chunk: bytes) -> dict[str, Any]:
        return {
            "type": "response.output_audio.delta",
            "response_id": "r1",
            "item_id": "i1",
            "output_index": 0,
            "content_index": 0,
            "delta": base64.b64encode(chunk).decode(),
        }

    out, _ = await _adapt(
        [
            {"type": "session.updated", "session": {"speak": {"format": {"rate": 16000}}}},
            delta(pcm[:5001]),
            delta(pcm[5001:]),
        ]
    )
    chunks = [base64.b64decode(e["delta"]) for e in out[1:]]
    assert all(len(c) % 2 == 0 for c in chunks)
    assert sum(len(c) for c in chunks) == pytest.approx(len(pcm) * 1.5, rel=0.1)


@pytest.mark.asyncio
async def test_events_reach_the_openai_realtime_handlers():
    service = _service()
    seen: list[str] = []

    async def _record(evt):
        seen.append(evt.type)

    for name in ("_handle_evt_session_created", "_handle_evt_conversation_item_added"):
        setattr(service, name, _record)

    async def _on_error(error: dict[str, Any], event_ids: set[str]):
        seen.append(f"error: {error['message']}")

    service._websocket = _ZoomAIRealtimeWebSocket(
        _FakeWebSocket(
            [
                {"type": "session.created", "session": {"listen": {}, "think": {}, "speak": {}}},
                {"type": "error", "error": {"message": "rejected"}},
                {
                    "type": "conversation.item.created",
                    "item": {"id": "i1", "type": "message", "role": "user", "content": []},
                },
            ]
        ),
        on_error=_on_error,
    )
    service._reconnects = 3  # no more reconnect attempts left
    service.push_error = _record_error = _ErrorRecorder()  # type: ignore[method-assign]
    await service._receive_task_handler()
    assert seen == ["session.created", "error: rejected", "conversation.item.added"]
    # The server closed the connection: the session is over.
    assert _record_error.messages == [
        "Zoom Voice Agent session ended: connection closed; reconnecting failed"
    ]
    assert service._websocket is None and not service._api_session_ready


@pytest.mark.asyncio
async def test_a_tool_result_is_created_right_after_its_call():
    service = _service()
    sent: list[Any] = []

    async def _send_client_event(event):
        sent.append(event)

    async def _noop(*args, **kwargs):
        pass

    service.send_client_event = _send_client_event  # type: ignore[method-assign]
    service._call_event_handler = _noop  # type: ignore[method-assign]
    call = openai_events.ConversationItem(
        id="fc_1", type="function_call", call_id="call_1", name="f", arguments="{}"
    )
    await service._handle_evt_conversation_item_added(
        openai_events.ConversationItemAdded(
            type="conversation.item.added", event_id="", previous_item_id="user_1", item=call
        )
    )
    service._open_function_calls.add("call_1")
    await service._send_tool_result("call_1", '{"ok": true}')
    (event,) = sent
    assert event.previous_item_id == "fc_1"
    assert event.item.type == "function_call_output" and event.item.call_id == "call_1"
    assert "call_1" not in service._open_function_calls


@pytest.mark.asyncio
async def test_later_updates_leave_out_the_model_and_audio_formats():
    # Only think and speak's voice and speed may change once the session is configured.
    service = _service(speak=events.Speak(voice="jenna"))
    service._listen_configured = True
    (session,) = await _session_updates(service)
    assert "provider" not in session["think"]
    assert session["speak"] == {"voice": "jenna"}


def test_listen_and_speak_settings_can_be_dicts():
    service = _service(
        listen={"transcription": {"language": "en", "vocabulary": {"phrases": ["Zoom"]}}},
        speak={"voice": "jenna"},
    )
    assert isinstance(service._settings.listen, events.Listen)
    assert service._settings.listen.transcription
    assert service._settings.listen.transcription.vocabulary
    assert service._settings.listen.transcription.vocabulary.phrases == ["Zoom"]
    assert isinstance(service._settings.speak, events.Speak)


@pytest.mark.asyncio
async def test_a_rejected_response_create_does_not_leave_a_response_active():
    service = _service()
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]
    sent: list[Any] = []

    async def _ws_send(message):
        sent.append(message)

    async def _push_frame(*args, **kwargs):
        pass

    service._ws_send = _ws_send  # type: ignore[method-assign]
    service.push_frame = _push_frame  # type: ignore[method-assign]
    service._response_active = True
    event = openai_events.ResponseCreateEvent(
        response=openai_events.ResponseProperties(output_modalities=["audio"])
    )
    await service.send_client_event(event)
    # no response object: the Voice Agent refuses one
    assert sent[-1] == {"type": "response.create", "event_id": event.event_id}

    await service._handle_server_error(
        {"type": "invalid_request_error", "message": "bad tools"}, {event.event_id}
    )
    assert not service._response_active
    assert errors.messages == ["Zoom Voice Agent error: bad tools"]


@pytest.mark.asyncio
async def test_a_response_create_refused_for_a_running_response_keeps_it_active():
    service = _service()
    service._pending_response_create = "evt_1"
    service._response_active = True
    await service._handle_server_error(
        {"message": "Conversation already has an active response"}, {"evt_1"}
    )
    assert service._response_active  # that response's response.done clears it
    assert service._response_create_owed  # and asks again if still needed


@pytest.mark.asyncio
async def test_a_failed_item_retrieve_is_passed_on():
    out, errors = await _adapt(
        [{"type": "error", "error": {"message": "no such item", "event_id": "rci_item_1"}}]
    )
    assert errors == []
    assert out[0]["error"]["code"] == "item_retrieve_invalid_item_id"
    openai_events.parse_server_event(json.dumps(out[0]))


@pytest.mark.asyncio
async def test_a_terminal_error_ends_the_session_with_one_error():
    service = _service()
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]

    async def _on_error(error, event_ids):
        pass

    service._websocket = _ZoomAIRealtimeWebSocket(
        _FakeWebSocket([{"type": "error", "error": {"code": "session_expired", "fatal": True}}]),
        on_error=_on_error,
    )
    await service._receive_task_handler()
    assert errors.messages == ["Zoom Voice Agent session ended: session_expired"]
    assert service._websocket is None


@pytest.mark.asyncio
async def test_an_error_names_its_client_event_in_the_error_or_the_event():
    ids: list[set[str]] = []

    async def _on_error(error, event_ids):
        ids.append(event_ids)

    ws = _ZoomAIRealtimeWebSocket(
        _FakeWebSocket(
            [
                {"type": "error", "event_id": "c1", "error": {"message": "x"}},
                {"type": "error", "error": {"message": "y", "event_id": "c2"}},
            ]
        ),
        on_error=_on_error,
    )
    assert [m async for m in ws] == []
    assert ids == [{"c1"}, {"c2"}]


@pytest.mark.asyncio
async def test_later_updates_leave_out_a_speak_format_and_text_only_speak():
    service = _service(speak=events.Speak(voice="jenna", format=events.AudioFormat(rate=16000)))
    service._listen_configured = True
    (session,) = await _session_updates(service)
    assert session["speak"] == {"voice": "jenna"}
    text_only = _service(
        session_properties=openai_events.SessionProperties(output_modalities=["text"])
    )
    text_only._listen_configured = True
    (session,) = await _session_updates(text_only)
    assert "speak" not in session


@pytest.mark.asyncio
async def test_a_retryable_end_reconnects_and_restores_the_conversation(monkeypatch):
    from pipecat.processors.aggregators.llm_context import LLMContext

    service = _service()
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]
    connects: list[int] = []

    async def _connect():
        connects.append(1)
        service._websocket = SimpleNamespace()  # type: ignore[assignment]

    async def _push_frame(*args, **kwargs):
        pass

    async def _on_error(error, event_ids):
        pass

    monkeypatch.setattr(llm_module, "_RECONNECT_BACKOFF_SECS", 0.0)

    service._connect = _connect  # type: ignore[method-assign]
    service.push_frame = _push_frame  # type: ignore[method-assign]
    service._response_active = True
    service._websocket = _ZoomAIRealtimeWebSocket(
        _FakeWebSocket(
            [{"type": "error", "error": {"code": "idle_timeout", "fatal": True, "retryable": True}}]
        ),
        on_error=_on_error,
    )
    await service._receive_task_handler()
    assert connects == [1] and errors.messages == []
    assert not service._response_active

    # In the new session, the conversation so far is created again.
    service._context = LLMContext(messages=[{"role": "user", "content": "My mug broke."}])
    sent: list[Any] = []

    async def _send_client_event(event):
        sent.append(event)

    service.send_client_event = _send_client_event  # type: ignore[method-assign]
    await service._handle_evt_session_updated(
        openai_events.SessionUpdatedEvent(
            type="session.updated", event_id="e1", session=openai_events.SessionProperties()
        )
    )
    assert [e.item.role for e in sent if hasattr(e, "item")] == ["user"]
    assert not service._llm_needs_conversation_setup
    # the reply the session ended in is asked for again
    assert any(isinstance(e, openai_events.ResponseCreateEvent) for e in sent)


@pytest.mark.asyncio
async def test_input_audio_is_resampled_and_sent_in_small_messages():
    service = _service()
    service._api_session_ready = True
    service._input_sample_rate = 16000
    sent: list[Any] = []

    async def _send_client_event(event):
        sent.append(event)

    service.send_client_event = _send_client_event  # type: ignore[method-assign]
    one_second_48k = b"\x01\x00" * 48000
    frame = InputAudioRawFrame(audio=one_second_48k, sample_rate=48000, num_channels=1)
    await service._send_user_audio(frame)
    chunks = [base64.b64decode(e.audio) for e in sent]
    assert all(len(json.dumps(e.model_dump(exclude_none=True))) < 32 * 1024 for e in sent)
    assert sum(len(c) for c in chunks) == pytest.approx(32000, rel=0.1)


@pytest.mark.asyncio
async def test_an_uncorrelated_refusal_of_a_response_is_recognized():
    # The dev server's errors don't name the client event they answer.
    service = _service()
    service.push_error = _ErrorRecorder()  # type: ignore[method-assign]

    async def _push_frame(*args, **kwargs):
        pass

    service.push_frame = _push_frame  # type: ignore[method-assign]
    service._pending_response_create = "evt_1"
    service._response_active = True
    await service._handle_server_error(
        {"message": "invalid client event: response.output_modalities.0: bad"}, {"event_x"}
    )
    assert not service._response_active


@pytest.mark.asyncio
async def test_reconnecting_gives_up_with_a_permanent_error(monkeypatch):
    service = _service()
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]
    connects: list[int] = []

    async def _connect():
        connects.append(1)  # the handshake fails: no websocket

    async def _on_error(error, event_ids):
        pass

    monkeypatch.setattr(llm_module, "_RECONNECT_BACKOFF_SECS", 0.0)
    service._connect = _connect  # type: ignore[method-assign]
    service._websocket = _ZoomAIRealtimeWebSocket(_FakeWebSocket([]), on_error=_on_error)
    await service._receive_task_handler()
    assert len(connects) == 3
    assert errors.messages == [
        "Zoom Voice Agent session ended: connection closed; reconnecting failed"
    ]


@pytest.mark.asyncio
async def test_a_completed_reply_restarts_the_reconnect_count():
    service = _service()
    service._reconnects = 2

    async def _push_frame(*args, **kwargs):
        pass

    service.push_frame = _push_frame  # type: ignore[method-assign]
    service._call_event_handler = _push_frame  # type: ignore[method-assign]
    await service._handle_evt_response_done(
        openai_events.ResponseDone.model_validate(
            {
                "type": "response.done",
                "event_id": "e",
                "response": {
                    "id": "r",
                    "object": "realtime.response",
                    "status": "completed",
                    "status_details": None,
                    "output": [],
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "input_token_details": {},
                        "output_token_details": {},
                    },
                },
            }
        )
    )
    assert service._reconnects == 0


@pytest.mark.asyncio
async def test_bursty_16khz_audio_is_resampled_without_losing_any(monkeypatch):
    """Deltas arrive in bursts with gaps over 0.2 s; none of the audio may be dropped."""
    import numpy as np
    import soxr

    import pipecat.audio.resamplers.soxr_stream_resampler as soxr_stream

    clock = {"t": 0.0}
    monkeypatch.setattr(soxr_stream.time, "time", lambda: clock["t"])
    tone = (np.sin(np.arange(16000 * 2) * 2 * np.pi * 220 / 16000) * 8000).astype(np.int16)
    pcm = tone.tobytes()
    cuts = [0, *range(4001, len(pcm), 4000), len(pcm)]  # ~125 ms each, cut mid-sample
    chunks = [pcm[a:b] for a, b in zip(cuts, cuts[1:])]

    class _ClockedWebSocket(_FakeWebSocket):
        def __aiter__(self):
            async def _gen():
                for index, m in enumerate(self._messages):
                    clock["t"] = index * 0.35  # a 350 ms gap before every message
                    yield m

            return _gen()

    ids = {"response_id": "r", "item_id": "i", "output_index": 0, "content_index": 0}
    messages = [
        {"type": "session.updated", "session": {"speak": {"format": {"rate": 16000}}}},
        {"type": "response.created", "response": {"id": "r"}},
        *(
            {"type": "response.output_audio.delta", **ids, "delta": base64.b64encode(c).decode()}
            for c in chunks
        ),
        {"type": "response.output_audio.done", **ids},
    ]

    async def _on_error(error, event_ids):
        pass

    ws = _ZoomAIRealtimeWebSocket(_ClockedWebSocket(messages), on_error=_on_error)
    out = [json.loads(m) async for m in ws]
    audio = b"".join(
        base64.b64decode(e["delta"]) for e in out if e["type"] == "response.output_audio.delta"
    )
    want = soxr.resample(tone, 16000, 24000, quality="VHQ")
    got = np.frombuffer(audio, dtype=np.int16)
    assert len(got) == len(want)  # nothing dropped at the gaps, the tail flushed at the end
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 2
    assert out[-1]["type"] == "response.output_audio.done"


# ---------------------------------------------------------------------------
# Pass-through settings, envelopes and the connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_is_merged_into_the_session_update():
    service = _service(
        extra={
            "think": {
                "provider": {"type": "openai", "model": "gpt-5", "headers": {"X-Api-Key": "k"}}
            },
            "listen": {"transcription": {"language": "en"}},
        }
    )
    (session,) = await _session_updates(service)
    assert session["think"]["provider"] == {
        "type": "openai",
        "model": "gpt-5",
        "headers": {"X-Api-Key": "k"},
    }
    assert session["listen"]["transcription"] == {"language": "en"}
    assert session["listen"]["format"] == {"type": "pcm16", "rate": 16000}


@pytest.mark.asyncio
async def test_extra_listen_fields_are_left_out_once_the_session_is_configured():
    service = _service(extra={"listen": {"transcription": {"language": "en"}}})
    service._listen_configured = True
    (session,) = await _session_updates(service)
    assert "listen" not in session


@pytest.mark.asyncio
async def test_your_own_provider_is_sent_with_the_model_in_the_first_update_only():
    service = _service(
        model="gpt-4.1-mini",
        provider=events.Provider(type="openai", headers={"X-Api-Key": "k"}),
    )
    (first,) = await _session_updates(service)
    assert first["think"]["provider"] == {
        "type": "openai",
        "model": "gpt-4.1-mini",
        "headers": {"X-Api-Key": "k"},
    }
    (later,) = await _session_updates(service)
    assert "provider" not in later["think"]


@pytest.mark.asyncio
async def test_a_providers_own_model_is_kept_and_a_dict_is_accepted():
    service = _service(
        provider={"type": "anthropic", "model": "claude-x", "headers": {"X-Api-Key": "k"}}
    )
    (session,) = await _session_updates(service)
    assert session["think"]["provider"] == {
        "type": "anthropic",
        "model": "claude-x",
        "headers": {"X-Api-Key": "k"},
    }


def test_provider_keys_are_left_out_of_reprs():
    provider = events.Provider(type="openai", headers={"X-Api-Key": "secret-key"})
    assert "secret-key" not in repr(provider)
    assert "secret-key" not in repr(ZoomAIRealtimeLLMService.Settings(provider=provider))


@pytest.mark.asyncio
async def test_a_rejected_provider_key_ends_the_session_without_reconnecting():
    service = _service()
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]
    connects: list[int] = []

    async def _connect():
        connects.append(1)

    async def _on_error(error, event_ids):
        pass

    service._connect = _connect  # type: ignore[method-assign]
    service._websocket = _ZoomAIRealtimeWebSocket(
        _FakeWebSocket(
            [
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_connection_credentials",
                        "message": "Provider rejected the BYOK credentials at setup",
                    },
                }
            ]
        ),
        on_error=_on_error,
    )
    await service._receive_task_handler()
    assert connects == []
    assert errors.messages == [
        "Zoom Voice Agent session ended: Provider rejected the BYOK credentials at setup"
    ]
    assert errors.kwargs[-1] == {"category": ErrorCategory.AUTHENTICATION}


@pytest.mark.asyncio
async def test_rejected_session_settings_are_a_permanent_invalid_request():
    service = _service(speak=events.Speak(voice="nosuchvoice"))
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]
    await _session_updates(service)  # the first update of the connection
    message = "invalid client event: session.audio.output.voice: Input should be 'alex'"
    await service._handle_server_error({"type": "invalid_request_error", "message": message}, set())
    assert errors.messages == [f"Zoom Voice Agent rejected the session settings: {message}"]
    assert errors.kwargs[-1] == {"category": ErrorCategory.INVALID_REQUEST}

    # Once the session is configured, a rejected update leaves it running.
    await service._handle_evt_session_updated(
        openai_events.SessionUpdatedEvent(
            type="session.updated", event_id="e1", session=openai_events.SessionProperties()
        )
    )
    await service._handle_server_error({"type": "invalid_request_error", "message": message}, set())
    assert errors.messages[-1] == f"Zoom Voice Agent error: {message}"
    assert errors.kwargs[-1] == {}


@pytest.mark.asyncio
async def test_rejected_session_settings_end_the_session_without_reconnecting():
    service = _service()
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]
    connects: list[int] = []

    async def _connect():
        connects.append(1)

    service._connect = _connect  # type: ignore[method-assign]
    service._configuring = True  # the first session update awaits session.updated
    service._websocket = _ZoomAIRealtimeWebSocket(
        _FakeWebSocket(
            [{"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}]
        ),
        on_error=service._handle_server_error,
    )
    await service._receive_task_handler()
    assert connects == []
    assert errors.messages == ["Zoom Voice Agent rejected the session settings: bad"]
    assert service._websocket is None


@pytest.mark.asyncio
async def test_enveloped_events_are_unwrapped():
    adapted, _ = await _adapt(
        [{"type": "session.updated", "payload": {"type": "session.updated", "session": {}}}]
    )
    assert [e["type"] for e in adapted] == ["session.updated"]
    assert "payload" not in adapted[0]


@pytest.mark.asyncio
async def test_16khz_agent_audio_is_resampled_by_default():
    # Without a session.updated naming the rate, agent audio is taken as 16 kHz.
    pcm = b"\x01\x00" * 1600  # 100 ms at 16 kHz
    adapted, _ = await _adapt(
        [
            {"type": "response.created", "response": {"id": "r1", "status": "in_progress"}},
            {
                "type": "response.output_audio.delta",
                "response_id": "r1",
                "item_id": "i1",
                "output_index": 0,
                "content_index": 0,
                "delta": base64.b64encode(pcm).decode(),
            },
        ]
    )
    delta = next(e for e in adapted if e["type"] == "response.output_audio.delta")
    assert len(base64.b64decode(delta["delta"])) > len(pcm)  # upsampled towards 24 kHz


@pytest.mark.asyncio
async def test_connect_sends_auth_tracking_id_and_subprotocol(monkeypatch):
    seen: dict[str, Any] = {}

    async def _fake_connect(*, uri, subprotocols, additional_headers, max_size):
        seen.update(uri=uri, subprotocols=subprotocols, headers=additional_headers)
        return _FakeWebSocket([])

    monkeypatch.setattr(llm_module, "websocket_connect", _fake_connect)
    service = _service()
    service.create_task = lambda coro, *a, **k: coro.close()  # type: ignore[method-assign]
    await service._connect()
    assert seen["uri"] == "wss://api.zoom.us/v2/ws/aiservices/voice-agent/live"
    assert seen["subprotocols"] == ["voice-agent.zoom.v2"]
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["headers"]["x-zm-trackingid"].startswith("zoom_ai_community_pipecat_")
    assert seen["headers"]["User-Agent"].startswith("pipecat/")


@pytest.mark.asyncio
async def test_a_refused_handshake_reports_zooms_reason(monkeypatch):
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    async def _refuse(**kwargs):
        raise InvalidStatus(Response(401, "Unauthorized", Headers(), b'{"message":"Invalid key"}'))

    monkeypatch.setattr(llm_module, "websocket_connect", _refuse)
    service = _service()
    service.push_error = errors = _ErrorRecorder()  # type: ignore[method-assign]
    await service._connect()
    assert errors.messages == ["Zoom refused the connection (HTTP 401): Invalid key"]
    assert re.fullmatch(
        r".* \(tracking_id=zoom_ai_community_pipecat_[0-9a-f-]{36}\)", errors.raw[0]
    )
    assert service._websocket is None


@pytest.mark.asyncio
async def test_a_call_with_empty_arguments_still_runs(monkeypatch):
    service = _service()
    ran: list[Any] = []

    async def _run_function_calls(calls):
        ran.extend(calls)

    service.run_function_calls = _run_function_calls  # type: ignore[method-assign]
    service._pending_function_calls["call_1"] = openai_events.ConversationItem(
        type="function_call", call_id="call_1", name="get_time", arguments=""
    )
    evt = openai_events.ResponseFunctionCallArgumentsDone(
        type="response.function_call_arguments.done",
        event_id="e1",
        response_id="r1",
        item_id="i1",
        output_index=0,
        call_id="call_1",
        arguments="",
    )
    await service._handle_evt_function_call_arguments_done(evt)
    assert [(c.function_name, c.arguments) for c in ran] == [("get_time", {})]
