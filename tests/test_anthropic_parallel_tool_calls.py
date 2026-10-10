#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for parallel tool calls in AnthropicLLMService streaming."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import FunctionCallFromLLM
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService


def _tool_use_block(tool_call_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_call_id, name=name)


def _parallel_tool_use_events() -> list[SimpleNamespace]:
    """A stream that requests two tool calls in one response.

    Anthropic streams each tool call as its own content block -- start,
    argument deltas, stop -- and reports the tool_use stop reason once, in a
    single terminal message_delta after every block has stopped.
    """
    return [
        SimpleNamespace(type="message_start"),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=_tool_use_block("toolu_01A", "get_weather"),
        ),
        SimpleNamespace(
            type="content_block_delta", index=0, delta=SimpleNamespace(partial_json='{"city":')
        ),
        SimpleNamespace(
            type="content_block_delta", index=0, delta=SimpleNamespace(partial_json='"Paris"}')
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=_tool_use_block("toolu_01B", "get_time"),
        ),
        SimpleNamespace(
            type="content_block_delta", index=1, delta=SimpleNamespace(partial_json='{"city":')
        ),
        SimpleNamespace(
            type="content_block_delta", index=1, delta=SimpleNamespace(partial_json='"Paris"}')
        ),
        SimpleNamespace(type="content_block_stop", index=1),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="tool_use")),
        SimpleNamespace(type="message_stop"),
    ]


async def _requested_function_calls() -> list[FunctionCallFromLLM]:
    """Stream two parallel tool calls and return the calls the service dispatches."""
    service = AnthropicLLMService(api_key="test-key")
    run_function_calls = AsyncMock()

    async def fake_stream(api_call, params):
        async def events():
            for event in _parallel_tool_use_events():
                yield event

        return events()

    async def drop_frame(frame, direction=None):
        pass

    with (
        patch.object(service, "push_frame", drop_frame),
        patch.object(service, "run_function_calls", run_function_calls),
        patch.object(service, "_create_message_stream", fake_stream),
    ):
        await service._process_context(LLMContext())

    run_function_calls.assert_awaited_once()
    (function_calls,) = run_function_calls.await_args.args
    return list(function_calls)


@pytest.mark.asyncio
async def test_every_parallel_tool_call_is_dispatched():
    """Both requested tool calls run."""
    function_calls = await _requested_function_calls()

    assert [(call.tool_call_id, call.function_name, call.arguments) for call in function_calls] == [
        ("toolu_01A", "get_weather", {"city": "Paris"}),
        ("toolu_01B", "get_time", {"city": "Paris"}),
    ]
