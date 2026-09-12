#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for frame emission in AzureVoiceLiveLLMService.

These tests drive the service's receive task handler with scripted Voice Live
server events and assert on the frames pushed downstream, using the event
names and payload shapes the live service sends.
"""

import base64
import json
from typing import Any

import pytest

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
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
from pipecat.services.azure.voicelive import events
from pipecat.services.azure.voicelive.llm import AzureVoiceLiveLLMService

RESPONSE_ID = "resp_1"
ITEM_ID = "msg_1"


def _make_service() -> AzureVoiceLiveLLMService:
    """Construct a service with no real connection. ``__init__`` does no I/O."""
    return AzureVoiceLiveLLMService(
        api_key="test-key",
        endpoint="https://my-resource.services.ai.azure.com",
    )


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
        self.frames: list[Any] = []

    async def __call__(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.frames.append(frame)

    def of_types(self, *types) -> list[Any]:
        return [f for f in self.frames if isinstance(f, types)]


class _BroadcastRecorder:
    """Records frame types broadcast via ``broadcast_frame``."""

    def __init__(self):
        self.types: list[Any] = []

    async def __call__(self, frame_type):
        self.types.append(frame_type)


async def _drive(service: AzureVoiceLiveLLMService, scripted: list[dict[str, Any]]) -> None:
    """Feed scripted server-event dicts through the receive handler."""
    service._websocket = _FakeWebSocket([json.dumps(e) for e in scripted])
    await service._receive_task_handler()


def _audio_delta(payload: bytes = b"\x00\x01\x02\x03") -> dict[str, Any]:
    return {
        "type": "response.audio.delta",
        "event_id": "event_audio",
        "response_id": RESPONSE_ID,
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "delta": base64.b64encode(payload).decode(),
    }


def _audio_done() -> dict[str, Any]:
    return {
        "type": "response.audio.done",
        "event_id": "event_audio_done",
        "response_id": RESPONSE_ID,
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
    }


def _response_done(status: str = "completed") -> dict[str, Any]:
    return {
        "type": "response.done",
        "event_id": "event_done",
        "response": {
            "object": "realtime.response",
            "id": RESPONSE_ID,
            "status": status,
            "status_details": None,
            "output": [],
            "usage": {"total_tokens": 10, "input_tokens": 4, "output_tokens": 6},
        },
    }


@pytest.mark.asyncio
async def test_audio_deltas_are_bracketed_by_tts_frames():
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    await _drive(service, [_audio_delta(), _audio_delta(), _audio_done(), _response_done()])

    started = recorder.of_types(TTSStartedFrame)
    stopped = recorder.of_types(TTSStoppedFrame)
    audio = recorder.of_types(TTSAudioRawFrame)

    assert len(started) == 1
    assert len(stopped) == 1
    assert len(audio) == 2

    started_idx = recorder.frames.index(started[0])
    stopped_idx = recorder.frames.index(stopped[0])
    assert all(started_idx < recorder.frames.index(f) < stopped_idx for f in audio)


@pytest.mark.asyncio
async def test_audio_frames_carry_the_configured_output_rate():
    service = _make_service()
    service._ensure_audio_config(16000, 16000)
    recorder = _FrameRecorder()
    service.push_frame = recorder

    await _drive(service, [_audio_delta()])

    audio = recorder.of_types(TTSAudioRawFrame)
    assert audio[0].sample_rate == 16000
    assert audio[0].num_channels == 1


@pytest.mark.asyncio
async def test_assistant_item_opens_and_response_done_closes_the_llm_response():
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    scripted = [
        {
            "type": "response.output_item.added",
            "event_id": "event_item",
            "response_id": RESPONSE_ID,
            "output_index": 0,
            "item": {
                "id": ITEM_ID,
                "object": "realtime.item",
                "type": "message",
                "status": "incomplete",
                "role": "assistant",
                "content": [],
            },
        },
        _response_done(),
    ]
    await _drive(service, scripted)

    assert len(recorder.of_types(LLMFullResponseStartFrame)) == 1
    assert len(recorder.of_types(LLMFullResponseEndFrame)) == 1


@pytest.mark.asyncio
async def test_audio_transcript_delta_pushes_tts_text():
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    scripted = [
        {
            "type": "response.audio_transcript.delta",
            "event_id": "event_t",
            "response_id": RESPONSE_ID,
            "item_id": ITEM_ID,
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }
    ]
    await _drive(service, scripted)

    text_frames = recorder.of_types(TTSTextFrame)
    assert [f.text for f in text_frames] == ["Hello"]


@pytest.mark.asyncio
async def test_input_transcription_deltas_accumulate_then_finalize():
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    scripted = [
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "event_id": "e1",
            "item_id": "item_u",
            "content_index": 0,
            "delta": "what's the ",
        },
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "event_id": "e2",
            "item_id": "item_u",
            "content_index": 0,
            "delta": "weather",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "e3",
            "item_id": "item_u",
            "content_index": 0,
            "transcript": "What's the weather?",
        },
    ]
    await _drive(service, scripted)

    interim = recorder.of_types(InterimTranscriptionFrame)
    final = recorder.of_types(TranscriptionFrame)

    assert [f.text for f in interim] == ["what's the ", "what's the weather"]
    assert [f.text for f in final] == ["What's the weather?"]
    assert service._interim_transcription_text == ""


@pytest.mark.asyncio
async def test_server_vad_speech_events_propose_user_turns():
    service = _make_service()
    service.push_frame = _FrameRecorder()
    broadcasts = _BroadcastRecorder()
    service.broadcast_frame = broadcasts

    scripted = [
        {"type": "input_audio_buffer.speech_started", "event_id": "e1", "audio_start_ms": 100},
        {"type": "input_audio_buffer.speech_stopped", "event_id": "e2", "audio_end_ms": 900},
    ]
    await _drive(service, scripted)

    assert broadcasts.types == [
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
    ]


@pytest.mark.asyncio
async def test_manual_turn_detection_suppresses_proposed_turn_frames():
    """With server VAD off, the caller's own strategy drives turn frames."""
    service = AzureVoiceLiveLLMService(
        api_key="test-key",
        endpoint="https://my-resource.services.ai.azure.com",
        settings=AzureVoiceLiveLLMService.Settings(
            session_properties=events.SessionProperties(turn_detection=None)
        ),
    )
    service.push_frame = _FrameRecorder()
    broadcasts = _BroadcastRecorder()
    service.broadcast_frame = broadcasts

    scripted = [
        {"type": "input_audio_buffer.speech_started", "event_id": "e1", "audio_start_ms": 100},
        {"type": "input_audio_buffer.speech_stopped", "event_id": "e2", "audio_end_ms": 900},
    ]
    await _drive(service, scripted)

    assert broadcasts.types == []


@pytest.mark.asyncio
async def test_non_fatal_error_does_not_end_the_receive_loop():
    """Cancelling with no active response is expected during interruptions."""
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    scripted = [
        {
            "type": "error",
            "event_id": "e1",
            "error": {
                "type": "invalid_request_error",
                "code": "response_cancel_not_active",
                "message": "No active response.",
            },
        },
        _audio_delta(),
    ]
    await _drive(service, scripted)

    assert len(recorder.of_types(TTSAudioRawFrame)) == 1


@pytest.mark.asyncio
async def test_an_assistant_item_opens_the_response_only_once():
    """The same item is announced by both conversation.item.created and
    response.output_item.added."""
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    item = {
        "id": ITEM_ID,
        "object": "realtime.item",
        "type": "message",
        "status": "incomplete",
        "role": "assistant",
        "content": [],
    }
    scripted = [
        {
            "type": "conversation.item.created",
            "event_id": "e1",
            "previous_item_id": "",
            "item": item,
        },
        {
            "type": "response.output_item.added",
            "event_id": "e2",
            "response_id": RESPONSE_ID,
            "output_index": 0,
            "item": item,
        },
        _response_done(),
    ]
    await _drive(service, scripted)

    assert len(recorder.of_types(LLMFullResponseStartFrame)) == 1
    assert len(recorder.of_types(LLMFullResponseEndFrame)) == 1


@pytest.mark.asyncio
async def test_interruption_closes_the_turn_only_once():
    """A cancelled response still reports done after the turn was closed."""
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    item = {
        "id": ITEM_ID,
        "object": "realtime.item",
        "type": "message",
        "status": "incomplete",
        "role": "assistant",
        "content": [],
    }
    await _drive(
        service,
        [
            {
                "type": "response.output_item.added",
                "event_id": "e1",
                "response_id": RESPONSE_ID,
                "output_index": 0,
                "item": item,
            },
            _audio_delta(),
        ],
    )

    await service._handle_interruption()
    await _drive(service, [_response_done(status="cancelled")])

    assert len(recorder.of_types(LLMFullResponseStartFrame)) == 1
    assert len(recorder.of_types(LLMFullResponseEndFrame)) == 1


@pytest.mark.asyncio
async def test_a_response_asked_for_mid_flight_waits_for_response_done():
    """The service rejects a second response while one is running."""
    service = _make_service()
    service.push_frame = _FrameRecorder()
    sent: list[Any] = []

    async def record(event):
        sent.append(type(event).__name__)

    service.send_client_event = record
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext([{"role": "user", "content": "hi"}])

    # A response is running.
    await _drive(service, [{"type": "response.created", "event_id": "e1", "response": {}}])
    assert service._response_in_flight is True

    # A tool result asks for the follow-up before that response finished.
    await service._create_response()
    assert "ResponseCreateEvent" not in sent
    assert service._run_llm_when_response_done is True

    # It goes out once the running response reports done.
    await _drive(service, [_response_done()])
    assert "ResponseCreateEvent" in sent
    assert service._run_llm_when_response_done is False


@pytest.mark.asyncio
async def test_an_interruption_drops_a_deferred_response():
    """The interrupting turn asks for its own response."""
    service = _make_service()
    service.push_frame = _FrameRecorder()
    sent: list[Any] = []

    async def record(event):
        sent.append(type(event).__name__)

    service.send_client_event = record
    service._api_session_ready = True
    service._response_in_flight = True
    service._run_llm_when_response_done = True

    await service._handle_interruption()

    assert service._run_llm_when_response_done is False

    await _drive(service, [_response_done(status="cancelled")])
    assert "ResponseCreateEvent" not in sent
