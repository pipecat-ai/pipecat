#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for typed input in OpenAI Realtime conversations."""

from typing import Any

import pytest

from pipecat.frames.frames import InputTextRawFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class _EventRecorder:
    """Records client events without opening a WebSocket."""

    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)


def _make_ready_service():
    service = OpenAIRealtimeLLMService(api_key="test-key")
    recorder = _EventRecorder()
    service.send_client_event = recorder  # type: ignore[method-assign]
    service._context = LLMContext()
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False

    async def _noop(*args, **kwargs):
        pass

    service.push_frame = _noop  # type: ignore[method-assign]
    service.start_processing_metrics = _noop  # type: ignore[method-assign]
    service.start_ttfb_metrics = _noop  # type: ignore[method-assign]
    return service, recorder


@pytest.mark.asyncio
async def test_typed_user_text_creates_item_and_response():
    service, recorder = _make_ready_service()

    await service.process_frame(
        InputTextRawFrame(text="Can you hear me?"), FrameDirection.DOWNSTREAM
    )

    assert [type(event) for event in recorder.events] == [
        events.ConversationItemCreateEvent,
        events.ResponseCreateEvent,
    ]
    item = recorder.events[0].item
    assert item.role == "user"
    assert item.content == [events.ItemContent(type="input_text", text="Can you hear me?")]
    assert service._messages_added_manually[item.id] is True


@pytest.mark.asyncio
async def test_typed_user_text_waits_for_initial_context_setup():
    service, recorder = _make_ready_service()
    service._api_session_ready = False

    await service._send_user_text("Hello")

    assert recorder.events == []
    assert service._run_llm_when_api_session_ready is True
