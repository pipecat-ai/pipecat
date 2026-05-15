#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for audio frame emission in OpenAIRealtimeLLMService.

These tests drive the service's receive task handler with scripted Realtime
API server events and assert on the frames pushed downstream. They cover
single-item responses (the common case) and multi-item responses (the
``gpt-realtime-2`` pattern), in both overlapping and non-overlapping
orderings of ``response.output_audio.done`` events.
"""

import base64
import json
from typing import Any

import pytest

from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_service() -> OpenAIRealtimeLLMService:
    """Construct a service with no real connection. ``__init__`` does no I/O."""
    return OpenAIRealtimeLLMService(api_key="test-key")


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


def _audio_delta(
    *,
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int = 0,
    audio_bytes: bytes = b"\x00" * 480,
) -> dict[str, Any]:
    """Build a ``response.output_audio.delta`` event dict."""
    return {
        "type": "response.output_audio.delta",
        "event_id": f"evt_delta_{item_id}_{len(audio_bytes)}",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "delta": base64.b64encode(audio_bytes).decode("ascii"),
    }


def _audio_done(
    *,
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int = 0,
) -> dict[str, Any]:
    """Build a ``response.output_audio.done`` event dict."""
    return {
        "type": "response.output_audio.done",
        "event_id": f"evt_done_{item_id}",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
    }


def _response_done(*, response_id: str) -> dict[str, Any]:
    """Build a ``response.done`` event dict with minimal valid usage stats."""
    return {
        "type": "response.done",
        "event_id": f"evt_response_done_{response_id}",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "completed",
            "status_details": None,
            "output": [],
            "usage": {
                "total_tokens": 10,
                "input_tokens": 5,
                "output_tokens": 5,
                "input_token_details": {
                    "cached_tokens": 0,
                    "text_tokens": 5,
                    "audio_tokens": 0,
                },
                "output_token_details": {
                    "text_tokens": 5,
                    "audio_tokens": 0,
                },
            },
        },
    }


class _FrameRecorder:
    """Records frames passed to a service's ``push_frame``."""

    def __init__(self):
        self.frames: list[Any] = []

    async def __call__(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.frames.append(frame)

    def of_types(self, *types) -> list[Any]:
        return [f for f in self.frames if isinstance(f, types)]


class _EventRecorder:
    """Records outgoing client events sent via ``send_client_event``."""

    def __init__(self):
        self.events: list[events.ClientEvent] = []

    async def __call__(self, event: events.ClientEvent):
        self.events.append(event)


async def _drive(service: OpenAIRealtimeLLMService, scripted: list[dict[str, Any]]) -> None:
    """Feed scripted server-event dicts through the receive handler."""
    service._websocket = _FakeWebSocket([json.dumps(e) for e in scripted])
    await service._receive_task_handler()


# ---------------------------------------------------------------------------
# Single-item happy path (regression guard — passes today, must keep passing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_item_response_unchanged():
    """A 1-item response emits exactly one Started/Stopped pair around audio."""
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    response_id = "resp_single"
    item_a = "item_A"

    scripted = [
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        _audio_done(response_id=response_id, item_id=item_a, output_index=0),
        _response_done(response_id=response_id),
    ]
    await _drive(service, scripted)

    started = recorder.of_types(TTSStartedFrame)
    stopped = recorder.of_types(TTSStoppedFrame)
    audio = recorder.of_types(TTSAudioRawFrame)

    assert len(started) == 1, "single-item should push exactly one TTSStartedFrame"
    assert len(stopped) == 1, "single-item should push exactly one TTSStoppedFrame"
    assert len(audio) == 2, "should push one TTSAudioRawFrame per delta"

    # Started precedes all audio; Stopped follows all audio.
    started_idx = recorder.frames.index(started[0])
    stopped_idx = recorder.frames.index(stopped[0])
    assert all(started_idx < recorder.frames.index(f) < stopped_idx for f in audio)

    # Tracking points at the single item.
    assert service._current_audio_response is not None
    assert service._current_audio_response.item_id == item_a


# ---------------------------------------------------------------------------
# Single-item interrupt (regression guard — passes today, must keep passing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_during_single_item_truncates_against_that_item():
    """Interrupting mid-stream sends a truncate event for the active item."""
    service = _make_service()
    service.push_frame = _FrameRecorder()
    sent = _EventRecorder()
    service.send_client_event = sent

    response_id = "resp_single_interrupt"
    item_a = "item_A"

    scripted = [
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
    ]
    await _drive(service, scripted)

    # Simulate interrupt mid-stream (before audio.done / response.done).
    await service._truncate_current_audio_response()

    truncates = [e for e in sent.events if isinstance(e, events.ConversationItemTruncateEvent)]
    assert len(truncates) == 1, "interrupt should send exactly one truncate event"
    assert truncates[0].item_id == item_a
    assert truncates[0].content_index == 0
    assert truncates[0].audio_end_ms >= 0


# ---------------------------------------------------------------------------
# Multi-item: overlapping audio.done (matches gpt-realtime-2 reproductions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_item_response_overlapping_emits_single_tts_turn():
    """Both audio.done events arrive after item B has started — one TTS pair."""
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    response_id = "resp_multi_overlap"
    item_a = "item_A"
    item_b = "item_B"

    scripted = [
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        # item B begins before item A's audio.done arrives
        _audio_delta(response_id=response_id, item_id=item_b, output_index=1),
        _audio_delta(response_id=response_id, item_id=item_b, output_index=1),
        _audio_done(response_id=response_id, item_id=item_a, output_index=0),
        _audio_done(response_id=response_id, item_id=item_b, output_index=1),
        _response_done(response_id=response_id),
    ]
    await _drive(service, scripted)

    started = recorder.of_types(TTSStartedFrame)
    stopped = recorder.of_types(TTSStoppedFrame)
    audio = recorder.of_types(TTSAudioRawFrame)

    assert len(started) == 1, "multi-item turn should emit exactly one TTSStartedFrame"
    assert len(stopped) == 1, "multi-item turn should emit exactly one TTSStoppedFrame"
    assert len(audio) == 4, "all four audio deltas should be forwarded in order"

    # Stopped is pushed on response.done — after both audio.done events.
    last_audio_done_idx = max(
        i for i, f in enumerate(recorder.frames) if isinstance(f, TTSAudioRawFrame)
    )
    stopped_idx = recorder.frames.index(stopped[0])
    assert stopped_idx > last_audio_done_idx, (
        "TTSStoppedFrame should be pushed at response.done, after all audio frames"
    )

    # All audio frames fall strictly between the bracketing pair.
    started_idx = recorder.frames.index(started[0])
    assert all(started_idx < recorder.frames.index(f) < stopped_idx for f in audio)

    # Last-item-wins truncation tracking.
    assert service._current_audio_response is not None
    assert service._current_audio_response.item_id == item_b


# ---------------------------------------------------------------------------
# Multi-item: non-overlapping audio.done (theoretical future-proofing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_item_response_non_overlapping_emits_single_tts_turn():
    """audio.done A arrives before item B begins — still one TTS pair."""
    service = _make_service()
    recorder = _FrameRecorder()
    service.push_frame = recorder

    response_id = "resp_multi_seq"
    item_a = "item_A"
    item_b = "item_B"

    scripted = [
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        _audio_done(response_id=response_id, item_id=item_a, output_index=0),
        _audio_delta(response_id=response_id, item_id=item_b, output_index=1),
        _audio_delta(response_id=response_id, item_id=item_b, output_index=1),
        _audio_done(response_id=response_id, item_id=item_b, output_index=1),
        _response_done(response_id=response_id),
    ]
    await _drive(service, scripted)

    started = recorder.of_types(TTSStartedFrame)
    stopped = recorder.of_types(TTSStoppedFrame)
    audio = recorder.of_types(TTSAudioRawFrame)

    assert len(started) == 1, "even with non-overlapping done events, only one Started"
    assert len(stopped) == 1, "even with non-overlapping done events, only one Stopped"
    assert len(audio) == 4

    started_idx = recorder.frames.index(started[0])
    stopped_idx = recorder.frames.index(stopped[0])
    assert all(started_idx < recorder.frames.index(f) < stopped_idx for f in audio)

    assert service._current_audio_response is not None
    assert service._current_audio_response.item_id == item_b


# ---------------------------------------------------------------------------
# Multi-item interrupt: truncates against the active (latest) item
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_during_second_item_truncates_against_second_item():
    """Interrupt during item B's audio sends truncate for item B (last-item-wins)."""
    service = _make_service()
    service.push_frame = _FrameRecorder()
    sent = _EventRecorder()
    service.send_client_event = sent

    response_id = "resp_multi_interrupt"
    item_a = "item_A"
    item_b = "item_B"

    scripted = [
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        _audio_delta(response_id=response_id, item_id=item_a, output_index=0),
        _audio_delta(response_id=response_id, item_id=item_b, output_index=1),
        _audio_delta(response_id=response_id, item_id=item_b, output_index=1),
    ]
    await _drive(service, scripted)

    await service._truncate_current_audio_response()

    truncates = [e for e in sent.events if isinstance(e, events.ConversationItemTruncateEvent)]
    assert len(truncates) == 1
    assert truncates[0].item_id == item_b, "must truncate the active (latest) item"
    assert truncates[0].content_index == 0
