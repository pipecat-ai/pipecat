#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for typed input in OpenAI Realtime conversations."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import LLMContextFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.tests.utils import run_test


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
    service._send_session_update = AsyncMock()
    return service, recorder


@pytest.mark.asyncio
async def test_typed_user_text_creates_item_and_response():
    service, recorder = _make_ready_service()
    message = {"role": "user", "content": "Can you hear me?"}
    service._context.add_message(message)
    await service.process_frame(
        LLMContextFrame(service._context, appended_messages=[message]), FrameDirection.DOWNSTREAM
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
@pytest.mark.parametrize("session_ready", [True, False])
async def test_first_text_is_seeded_once(session_ready):
    service, recorder = _make_ready_service()
    service._context = None
    service._api_session_ready = session_ready
    service._llm_needs_conversation_setup = True
    message = {"role": "user", "content": "Hello"}
    context = LLMContext(messages=[message])
    await service.process_frame(
        LLMContextFrame(context, appended_messages=[message]), FrameDirection.DOWNSTREAM
    )
    if not session_ready:
        assert recorder.events == []
        assert service._run_llm_when_api_session_ready
        service._api_session_ready = True
        await service._create_response()
    assert [type(event) for event in recorder.events] == [
        events.ConversationItemCreateEvent,
        events.ResponseCreateEvent,
    ]
    assert recorder.events[0].item.content[0].text == "Hello"
    assert context.get_messages() == [message]


@pytest.mark.asyncio
async def test_identical_consecutive_messages_are_each_delivered_once():
    service, recorder = _make_ready_service()
    for _ in range(2):
        message = {"role": "user", "content": "Again"}
        service._context.add_message(message)
        await service.process_frame(
            LLMContextFrame(service._context, appended_messages=[message]),
            FrameDirection.DOWNSTREAM,
        )
    assert [type(event) for event in recorder.events] == [
        events.ConversationItemCreateEvent,
        events.ResponseCreateEvent,
    ] * 2


@pytest.mark.asyncio
async def test_ordinary_context_update_does_not_replay_user_history():
    service, recorder = _make_ready_service()
    service._context.add_message({"role": "user", "content": "Already sent"})
    await service.process_frame(LLMContextFrame(service._context), FrameDirection.DOWNSTREAM)
    assert recorder.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregator_type", [LLMUserAggregator, LLMAssistantAggregator])
async def test_deferred_input_is_delivered_once_on_the_next_context_push(aggregator_type):
    service, recorder = _make_ready_service()
    aggregator = aggregator_type(service._context)

    async def deliver(frame, direction=FrameDirection.DOWNSTREAM):
        await service.process_frame(frame, direction)

    aggregator.push_frame = deliver
    deferred = {"role": "user", "content": "Remember this"}
    await aggregator._handle_llm_messages_append(LLMMessagesAppendFrame(messages=[deferred]))
    assert recorder.events == []
    immediate = {"role": "user", "content": "Answer now"}
    await aggregator._handle_llm_messages_append(
        LLMMessagesAppendFrame(messages=[immediate], run_llm=True)
    )
    assert [type(event) for event in recorder.events] == [
        events.ConversationItemCreateEvent,
        events.ConversationItemCreateEvent,
        events.ResponseCreateEvent,
    ]
    assert [event.item.content[0].text for event in recorder.events[:2]] == [
        "Remember this",
        "Answer now",
    ]
    assert service._context.get_messages() == [deferred, immediate]
    recorder.events.clear()
    await aggregator.push_context_frame()
    assert recorder.events == []
    await aggregator.cleanup()


@pytest.mark.asyncio
async def test_replaced_context_does_not_deliver_obsolete_pending_input():
    service, recorder = _make_ready_service()
    aggregator = LLMUserAggregator(service._context)
    aggregator.push_frame = AsyncMock()
    await aggregator._handle_llm_messages_append(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Obsolete"}])
    )
    aggregator.set_messages([])
    await aggregator.push_context_frame()
    frame = aggregator.push_frame.call_args.args[0]
    assert frame.appended_messages == []
    assert frame.context.get_messages() == []
    await aggregator.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("initialized", [True, False])
async def test_queued_text_updates_are_delivered_once_in_order(initialized):
    service, recorder = _make_ready_service()
    service._connect = AsyncMock()
    service._disconnect = AsyncMock()
    # Restore the real push path so the test exercises pipeline queues.
    del service.push_frame
    context = service._context
    if not initialized:
        service._context = None
        service._llm_needs_conversation_setup = True
    await run_test(
        Pipeline([LLMUserAggregator(context), service]),
        frames_to_send=[
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True)
            for text in ("First", "Second")
        ],
    )
    items = [
        event.item
        for event in recorder.events
        if isinstance(event, events.ConversationItemCreateEvent)
    ]
    assert [item.content[0].text for item in items] == ["First", "Second"]
    assert (
        len([event for event in recorder.events if isinstance(event, events.ResponseCreateEvent)])
        == 2
    )
