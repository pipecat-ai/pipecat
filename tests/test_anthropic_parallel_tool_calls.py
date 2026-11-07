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


def _usage() -> SimpleNamespace:
    """The usage envelope the service reads from message_start/message_delta."""
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=20,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        output_tokens_details=SimpleNamespace(thinking_tokens=0),
    )


def _parallel_tool_use_events() -> list[SimpleNamespace]:
    """A stream that requests two tool calls in one response.

    Anthropic streams each tool call as its own content block -- start,
    argument deltas, stop -- and reports the tool_use stop reason once, in a
    single terminal message_delta after every block has stopped.
    """
    return [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=_usage())),
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
        SimpleNamespace(
            type="message_delta", delta=SimpleNamespace(stop_reason="tool_use"), usage=_usage()
        ),
        SimpleNamespace(type="message_stop"),
    ]


def _truncated_tool_use_events() -> list[SimpleNamespace]:
    """A stream cut off by max_tokens: the first tool call is complete, the
    second block's arguments are truncated mid-JSON, and the terminal
    stop_reason is max_tokens, not tool_use."""
    return [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=_usage())),
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
            type="content_block_delta", index=1, delta=SimpleNamespace(partial_json='{"city')
        ),
        SimpleNamespace(type="content_block_stop", index=1),
        SimpleNamespace(
            type="message_delta", delta=SimpleNamespace(stop_reason="max_tokens"), usage=_usage()
        ),
        SimpleNamespace(type="message_stop"),
    ]


async def _dispatched_function_calls(events) -> list[FunctionCallFromLLM]:
    """Stream `events` through the service and return the calls it dispatches."""
    service = AnthropicLLMService(api_key="test-key")
    run_function_calls = AsyncMock()

    async def fake_stream(api_call, params):
        async def stream_events():
            for event in events:
                yield event

        return stream_events()

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
    function_calls = await _dispatched_function_calls(_parallel_tool_use_events())

    assert [(call.tool_call_id, call.function_name, call.arguments) for call in function_calls] == [
        ("toolu_01A", "get_weather", {"city": "Paris"}),
        ("toolu_01B", "get_time", {"city": "Paris"}),
    ]


@pytest.mark.asyncio
async def test_stream_cut_by_max_tokens_dispatches_no_calls():
    """A response that stopped on max_tokens executes none of its tool calls.

    The stream contains one complete tool call and one truncated one;
    dispatching either would run tool input Anthropic reports as incomplete.
    """
    function_calls = await _dispatched_function_calls(_truncated_tool_use_events())

    assert function_calls == []


@pytest.mark.parametrize(
    "delta",
    [SimpleNamespace(stop_reason=None), SimpleNamespace()],
    ids=["stop-reason-none", "stop-reason-omitted"],
)
@pytest.mark.asyncio
async def test_interim_message_delta_keeps_pending_tools(delta):
    """An interim message_delta without a stop reason keeps collected calls.

    The stream contract allows more than one message_delta, and the SDK's
    optional stop_reason is present-but-None on deltas that don't carry one,
    so only a delta that actually reports a stop reason may consume the
    collected calls.
    """
    events = _parallel_tool_use_events()
    events.insert(-2, SimpleNamespace(type="message_delta", delta=delta, usage=_usage()))

    function_calls = await _dispatched_function_calls(events)

    assert [(call.tool_call_id, call.function_name, call.arguments) for call in function_calls] == [
        ("toolu_01A", "get_weather", {"city": "Paris"}),
        ("toolu_01B", "get_time", {"city": "Paris"}),
    ]
