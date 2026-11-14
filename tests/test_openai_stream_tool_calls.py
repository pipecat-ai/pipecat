#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that the OpenAI-compatible streaming loop assembles every streamed tool call.

A delta can carry several tool calls, and some OpenAI-compatible servers reuse an
index for different calls or start at an index other than 0. Each call has to come
out whole, with its own id, name and arguments.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService


def _call(index, id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=function)


def _chunk(*tool_calls):
    delta = SimpleNamespace(tool_calls=list(tool_calls), content=None)
    return SimpleNamespace(
        usage=None, model=None, choices=[SimpleNamespace(delta=delta, finish_reason=None)]
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


async def _run(chunks):
    """Stream the chunks through the service and return the function calls it ran."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            api_key="test-key", settings=OpenAILLMService.Settings(model="test-model")
        )
    service._client = AsyncMock()
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.run_function_calls = AsyncMock()

    await service._process_context(LLMContext(messages=[{"role": "user", "content": "Hi"}]))

    if not service.run_function_calls.called:
        return []
    return [
        (call.tool_call_id, call.function_name, call.arguments)
        for call in service.run_function_calls.call_args.args[0]
    ]


@pytest.mark.asyncio
async def test_one_call_split_across_chunks():
    calls = await _run(
        [
            _chunk(_call(0, "call_a", "get_weather", "")),
            _chunk(_call(0, arguments='{"city": ')),
            _chunk(_call(0, arguments='"Paris"}')),
        ]
    )
    assert calls == [("call_a", "get_weather", {"city": "Paris"})]


@pytest.mark.asyncio
async def test_parallel_calls_with_their_own_indexes():
    calls = await _run(
        [
            _chunk(_call(0, "call_a", "get_weather", '{"city": "Paris"}')),
            _chunk(_call(1, "call_b", "get_time", "")),
            _chunk(_call(1, arguments='{"tz": "UTC"}')),
        ]
    )
    assert calls == [
        ("call_a", "get_weather", {"city": "Paris"}),
        ("call_b", "get_time", {"tz": "UTC"}),
    ]


@pytest.mark.asyncio
async def test_two_calls_in_one_chunk_are_both_kept():
    calls = await _run(
        [
            _chunk(
                _call(0, "call_a", "get_weather", '{"city": "Paris"}'),
                _call(1, "call_b", "get_time", '{"tz": "UTC"}'),
            )
        ]
    )
    assert calls == [
        ("call_a", "get_weather", {"city": "Paris"}),
        ("call_b", "get_time", {"tz": "UTC"}),
    ]


@pytest.mark.asyncio
async def test_parallel_calls_reusing_index_zero_stay_separate():
    calls = await _run(
        [
            _chunk(_call(0, "call_a", "get_weather", '{"city": "Paris"}')),
            _chunk(_call(0, "call_b", "get_time", '{"tz": "UTC"}')),
        ]
    )
    assert calls == [
        ("call_a", "get_weather", {"city": "Paris"}),
        ("call_b", "get_time", {"tz": "UTC"}),
    ]


@pytest.mark.asyncio
async def test_first_call_at_index_one_does_not_add_an_empty_call():
    calls = await _run([_chunk(_call(1, "call_a", "get_weather", '{"city": "Paris"}'))])
    assert calls == [("call_a", "get_weather", {"city": "Paris"})]


@pytest.mark.asyncio
async def test_call_without_arguments_runs_with_an_empty_object():
    calls = await _run([_chunk(_call(0, "call_a", "get_time", None))])
    assert calls == [("call_a", "get_time", {})]


@pytest.mark.asyncio
async def test_no_tool_calls_run_nothing():
    assert await _run([]) == []
