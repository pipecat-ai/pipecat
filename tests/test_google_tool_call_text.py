#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for text handling around streamed Google (Gemini) tool calls.

Suppression is centralized in ``LLMService._push_llm_text``; these tests verify
the Google streaming loop signals detection at the right time and that the
end-to-end frame flow matches the configured policy.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.settings import ToolCallTextPolicy


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


def _text_part(text):
    return SimpleNamespace(
        text=text, thought=False, function_call=None, inline_data=None, thought_signature=None
    )


def _function_call_part(*, name="record_name", call_id="fc_1", args=None):
    return SimpleNamespace(
        text=None,
        thought=False,
        function_call=SimpleNamespace(id=call_id, name=name, args=args or {"name": "Sam"}),
        inline_data=None,
        thought_signature=None,
    )


def _chunk(parts):
    return SimpleNamespace(
        usage_metadata=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
                finish_reason=None,
                grounding_metadata=None,
            )
        ],
    )


def _service(chunks, *, policy=ToolCallTextPolicy.PRESERVE):
    with patch.object(GoogleLLMService, "create_client"):
        service = GoogleLLMService(
            api_key="test-key",
            settings=GoogleLLMService.Settings(
                model="gemini-3.6-flash",
                tool_call_text_policy=policy,
            ),
        )
    service._stream_response = lambda ctx: _FakeStream(chunks)
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

    with (
        patch.object(FrameProcessor, "push_frame", fake_push),
        patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()),
    ):
        await service._process_context(context)
    return pushed


@pytest.mark.asyncio
async def test_google_text_is_preserved_by_default_around_a_tool_call():
    service = _service(
        [
            _chunk([_text_part("Before."), _function_call_part()]),
            _chunk([_text_part("After.")]),
        ]
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Before.", "After."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_google_text_after_a_tool_call_can_be_suppressed():
    service = _service(
        [
            _chunk([_text_part("Before."), _function_call_part()]),
            _chunk([_text_part("After.")]),
        ],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL_DETECTED,
    )

    pushed = await _run_and_collect(service, _context())

    assert pushed == ["Before."]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_google_suppression_state_resets_for_the_next_response():
    service = _service(
        [
            _chunk([_function_call_part()]),
            _chunk([_text_part("Suppressed.")]),
        ],
        policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL_DETECTED,
    )

    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with (
        patch.object(FrameProcessor, "push_frame", fake_push),
        patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()),
    ):
        await service._process_context(_context())
        assert pushed == []

        service._stream_response = lambda ctx: _FakeStream([_chunk([_text_part("Final.")])])
        await service._process_context(_context())

    assert pushed == ["Final."]
