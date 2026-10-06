#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAIRealtimeLLMService carrying a conversation past session expiry.

OpenAI ends a Realtime session after a fixed maximum duration (60 minutes) with
a ``session_expired`` error. The service should reconnect and reseed the new
session from its local context instead of treating the error as fatal.
"""

import json
from typing import Any

import pytest

from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    TTSStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import CurrentAudioResponse, OpenAIRealtimeLLMService


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


def _session_expired() -> dict[str, Any]:
    return {
        "type": "error",
        "event_id": "evt_session_expired",
        "error": {
            "type": "invalid_request_error",
            "code": "session_expired",
            "message": "Your session hit the maximum duration of 60 minutes.",
            "param": None,
            "event_id": None,
        },
    }


def _make_service():
    service = OpenAIRealtimeLLMService(api_key="test-key")
    pushed: list[Any] = []
    sent: list[events.ClientEvent] = []
    scheduled: list[Any] = []

    async def push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    async def send_client_event(event):
        sent.append(event)

    def create_task(coroutine, name=None, context=None):
        scheduled.append(coroutine)

    service.push_frame = push_frame
    service.send_client_event = send_client_event
    service.create_task = create_task
    return service, pushed, sent, scheduled


@pytest.mark.asyncio
async def test_session_expired_reconnects_instead_of_failing():
    service, pushed, _, scheduled = _make_service()
    resets = []

    async def reset_conversation():
        resets.append(True)

    service.reset_conversation = reset_conversation
    # The session expires in the middle of a spoken response.
    service._current_assistant_response = object()
    service._current_audio_response = CurrentAudioResponse(
        item_id="item_1", content_index=0, start_time_ms=0
    )

    service._websocket = _FakeWebSocket([json.dumps(_session_expired())])
    await service._receive_task_handler()

    assert not any(isinstance(f, ErrorFrame) for f in pushed)
    # The reconnect can't run on the receive task, which it cancels.
    assert len(scheduled) == 1
    await scheduled[0]

    assert resets == [True]
    assert [type(f) for f in pushed] == [TTSStoppedFrame, LLMFullResponseEndFrame]
    assert service._current_assistant_response is None
    assert service._current_audio_response is None


@pytest.mark.asyncio
async def test_new_session_is_seeded_from_context_without_a_response():
    service, _, sent, _ = _make_service()
    service._context = LLMContext(
        messages=[
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": "Nice to meet you, Ada."},
        ]
    )
    # State after reset_conversation(): the new session hasn't seen the conversation.
    service._llm_needs_conversation_setup = True

    await service._handle_evt_session_updated(None)

    items = [e for e in sent if isinstance(e, events.ConversationItemCreateEvent)]
    assert items
    assert "Ada" in json.dumps([e.model_dump() for e in items])
    assert not any(isinstance(e, events.ResponseCreateEvent) for e in sent)
    assert service._llm_needs_conversation_setup is False


@pytest.mark.asyncio
async def test_session_updated_does_not_seed_before_first_context():
    service, _, sent, _ = _make_service()

    await service._handle_evt_session_updated(None)

    assert sent == []
    assert service._llm_needs_conversation_setup is True
