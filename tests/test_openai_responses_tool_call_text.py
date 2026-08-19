#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for text handling around streamed OpenAI Responses API tool calls.

Suppression is centralized in ``LLMService._push_llm_text``; these tests verify
the Responses HTTP streaming loop signals detection at the right time and that
the end-to-end frame flow matches the configured policy.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
)

from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService
from pipecat.services.settings import ToolCallTextPolicy


class _FakeAsyncStream:
    def __init__(self, events):
        self._events = list(events)

    async def _iterator(self):
        for event in self._events:
            yield event

    def __aiter__(self):
        return self._iterator()

    async def close(self):
        pass


def _text_delta(text):
    event = MagicMock(spec=ResponseTextDeltaEvent)
    event.delta = text
    return event


def _tool_call_added(*, item_id="fc_1", call_id="call_1", name="lookup"):
    item = MagicMock(spec=ResponseFunctionToolCall)
    item.id = item_id
    item.call_id = call_id
    item.name = name
    item.arguments = ""
    event = MagicMock(spec=ResponseOutputItemAddedEvent)
    event.item = item
    return event


def _tool_call_args_done(*, item_id="fc_1", arguments='{"q":"test"}'):
    event = MagicMock(spec=ResponseFunctionCallArgumentsDoneEvent)
    event.item_id = item_id
    event.arguments = arguments
    return event


def _tool_call_done(*, item_id="fc_1", call_id="call_1", name="lookup", arguments='{"q":"test"}'):
    item = MagicMock(spec=ResponseFunctionToolCall)
    item.id = item_id
    item.call_id = call_id
    item.name = name
    item.arguments = arguments
    event = MagicMock(spec=ResponseOutputItemDoneEvent)
    event.item = item
    return event


def _completed():
    response = MagicMock()
    response.usage = None
    response.model = "gpt-4.1"
    response.output = []
    event = MagicMock(spec=ResponseCompletedEvent)
    event.response = response
    return event


def _service(*, policy=ToolCallTextPolicy.PRESERVE):
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        service = OpenAIResponsesHttpLLMService(
            api_key="test-key",
            settings=OpenAIResponsesHttpLLMService.Settings(
                model="gpt-4.1",
                tool_call_text_policy=policy,
            ),
        )
    service._client = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_llm_usage_metrics = AsyncMock()
    service.run_function_calls = AsyncMock()

    adapter = MagicMock()
    adapter.get_messages_for_logging.return_value = []
    adapter.get_llm_invocation_params.return_value = {}
    service.get_llm_adapter = MagicMock(return_value=adapter)
    service._build_response_params = MagicMock(return_value={})
    return service


def _context():
    return LLMContext(messages=[{"role": "user", "content": "Look it up"}])


async def _run_and_collect(service, events):
    service._create_stream = AsyncMock(return_value=_FakeAsyncStream(events))

    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await service._process_context(_context())
    return pushed


@pytest.mark.asyncio
async def test_responses_text_is_preserved_by_default_around_a_tool_call():
    service = _service()
    pushed = await _run_and_collect(
        service,
        [
            _text_delta("Before."),
            _tool_call_added(),
            _tool_call_args_done(),
            _tool_call_done(),
            _text_delta("After."),
            _completed(),
        ],
    )
    assert pushed == ["Before.", "After."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_text_after_a_tool_call_can_be_suppressed():
    service = _service(policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL)
    pushed = await _run_and_collect(
        service,
        [
            _text_delta("Before."),
            _tool_call_added(),
            _tool_call_args_done(),
            _tool_call_done(),
            _text_delta("After."),
            _completed(),
        ],
    )
    assert pushed == ["Before."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_suppression_state_resets_for_the_next_response():
    service = _service(policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL)
    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        service._create_stream = AsyncMock(
            return_value=_FakeAsyncStream(
                [_tool_call_added(), _tool_call_args_done(), _tool_call_done(), _completed()]
            )
        )
        await service._process_context(_context())
        assert pushed == []

        await service.push_frame(LLMFullResponseStartFrame())
        service._create_stream = AsyncMock(
            return_value=_FakeAsyncStream([_text_delta("Final."), _completed()])
        )
        await service._process_context(_context())

    assert pushed == ["Final."]
