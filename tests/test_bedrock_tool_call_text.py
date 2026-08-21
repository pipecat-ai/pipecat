#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for text handling around streamed Bedrock tool calls.

Suppression is centralized in ``LLMService._push_llm_text``; these tests verify
the Bedrock streaming loop signals detection at the right time and that the
end-to-end frame flow matches the configured policy.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.settings import ToolCallTextPolicy


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield event


def _text(text):
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}


def _tool_start():
    return {
        "contentBlockStart": {
            "contentBlockIndex": 1,
            "start": {"toolUse": {"toolUseId": "tool_1", "name": "record_name"}},
        }
    }


def _tool_args():
    return {
        "contentBlockDelta": {
            "contentBlockIndex": 1,
            "delta": {"toolUse": {"input": '{"name":"Sam"}'}},
        }
    }


def _tool_stop():
    return {"contentBlockStop": {"contentBlockIndex": 1}}


def _service(events, *, policy=ToolCallTextPolicy.PRESERVE):
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(
            model="us.amazon.nova-lite-v1:0",
            tool_call_text_policy=policy,
        )
    )

    @asynccontextmanager
    async def fake_client(*args, **kwargs):
        yield object()

    service._aws_session.create_client = fake_client
    service._create_converse_stream = AsyncMock(return_value={"stream": _FakeStream(events)})
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.run_function_calls = AsyncMock()
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
async def test_bedrock_text_is_preserved_by_default_after_a_tool_call():
    service = _service([_tool_start(), _tool_args(), _tool_stop(), _text("Recorded.")])

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Recorded."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_bedrock_text_after_a_tool_call_can_be_suppressed():
    service = _service(
        [_text("Before."), _tool_start(), _tool_args(), _tool_stop(), _text("After.")],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Before."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_bedrock_suppression_state_resets_for_the_next_response():
    service = _service(
        [_tool_start(), _tool_args(), _tool_stop(), _text("Suppressed.")],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL,
    )

    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await service._process_context(_context())
        assert pushed == []

        await service.push_frame(LLMFullResponseStartFrame())
        service._create_converse_stream = AsyncMock(
            return_value={"stream": _FakeStream([_text("Final.")])}
        )
        await service._process_context(_context())

    assert pushed == ["Final."]
