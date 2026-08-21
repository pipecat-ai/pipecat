#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for text handling around streamed Anthropic tool calls.

Suppression is centralized in ``LLMService._push_llm_text``; these tests verify
the Anthropic streaming loop signals detection at the right time and that the
end-to-end frame flow matches the configured policy.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.settings import ToolCallTextPolicy


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield event


def _text_delta(text):
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text=text))


def _tool_use_start(*, name="record_name", tool_id="tool_1"):
    return SimpleNamespace(
        type="content_block_start",
        content_block=SimpleNamespace(type="tool_use", id=tool_id, name=name),
    )


def _partial_json(fragment):
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(partial_json=fragment))


def _message_delta_tool_use():
    return SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="tool_use"))


def _service(events, *, policy=ToolCallTextPolicy.PRESERVE):
    fake_client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))
    )
    service = AnthropicLLMService(
        api_key="test-key",
        client=fake_client,
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-4-6",
            tool_call_text_policy=policy,
        ),
    )
    service._create_message_stream = AsyncMock(return_value=_FakeStream(events))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.run_function_calls = AsyncMock()
    service._report_usage_metrics = AsyncMock()
    return service


def _context():
    return LLMContext(
        messages=[{"role": "user", "content": "Record Sam"}],
        tools=[
            FunctionSchema(
                name="record_name",
                description="Record a supplied first name.",
                properties={"name": {"type": "string"}},
                required=["name"],
            )
        ],
    )


async def _run_and_collect(service, context):
    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await service._process_context(context)
    return pushed


@pytest.mark.asyncio
async def test_anthropic_text_is_preserved_by_default_around_a_tool_call():
    service = _service(
        [
            _text_delta("Before."),
            _tool_use_start(),
            _partial_json('{"name":"Sam"}'),
            _message_delta_tool_use(),
            _text_delta("After."),
        ]
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Before.", "After."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_text_after_a_tool_call_can_be_suppressed():
    service = _service(
        [
            _text_delta("Before."),
            _tool_use_start(),
            _partial_json('{"name":"Sam"}'),
            _message_delta_tool_use(),
            _text_delta("After."),
        ],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Before."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_suppression_state_resets_for_the_next_response():
    service = _service(
        [
            _tool_use_start(),
            _partial_json('{"name":"Sam"}'),
            _message_delta_tool_use(),
            _text_delta("Suppressed."),
        ],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
    )

    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await service._process_context(_context())
        assert pushed == []

        service._create_message_stream = AsyncMock(
            return_value=_FakeStream([_text_delta("Final.")])
        )
        await service._process_context(_context())

    assert pushed == ["Final."]
