#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for streaming tool-call aggregation in the OpenAI base LLM service."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService


def _tool_call_chunk(index, id=None, name=None, arguments=None):
    function = None
    if name is not None or arguments is not None:
        function = SimpleNamespace(name=name, arguments=arguments)
    tool_call = SimpleNamespace(index=index, id=id, function=function)
    delta = SimpleNamespace(tool_calls=[tool_call], content=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(usage=None, model=None, choices=[choice])


async def _run_tool_call_chunks(chunks):
    """Feed synthetic streaming chunks through _process_context and capture the calls."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        async def chunk_stream():
            for chunk in chunks:
                yield chunk

        service.get_chat_completions = AsyncMock(return_value=chunk_stream())
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        captured = []

        async def capture_function_calls(function_calls):
            captured.extend(function_calls)

        service.run_function_calls = capture_function_calls

        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        await service._process_context(context)
        return captured


@pytest.mark.asyncio
async def test_tool_call_id_captured_when_id_and_name_arrive_in_separate_deltas():
    """The tool-call id must be captured even when it arrives on a delta without the name."""
    calls = await _run_tool_call_chunks(
        [
            _tool_call_chunk(index=0, id="call_abc123"),
            _tool_call_chunk(index=0, name="get_weather"),
            _tool_call_chunk(index=0, arguments='{"location": "Paris"}'),
        ]
    )
    assert len(calls) == 1
    assert calls[0].tool_call_id == "call_abc123"
    assert calls[0].function_name == "get_weather"
    assert calls[0].arguments == {"location": "Paris"}


@pytest.mark.asyncio
async def test_tool_call_id_captured_when_id_and_name_arrive_together():
    """The common case (id and name on the same delta) keeps working."""
    calls = await _run_tool_call_chunks(
        [
            _tool_call_chunk(index=0, id="call_abc123", name="get_weather"),
            _tool_call_chunk(index=0, arguments='{"location": "Paris"}'),
        ]
    )
    assert len(calls) == 1
    assert calls[0].tool_call_id == "call_abc123"
    assert calls[0].function_name == "get_weather"


@pytest.mark.asyncio
async def test_tool_call_ids_captured_for_multiple_tool_calls_with_split_deltas():
    """Each parallel tool call keeps its own id when ids and names are split across deltas."""
    calls = await _run_tool_call_chunks(
        [
            _tool_call_chunk(index=0, id="call_first"),
            _tool_call_chunk(index=0, name="get_weather"),
            _tool_call_chunk(index=0, arguments='{"location": "Paris"}'),
            _tool_call_chunk(index=1, id="call_second"),
            _tool_call_chunk(index=1, name="get_time"),
            _tool_call_chunk(index=1, arguments='{"timezone": "CET"}'),
        ]
    )
    assert len(calls) == 2
    assert calls[0].tool_call_id == "call_first"
    assert calls[0].function_name == "get_weather"
    assert calls[1].tool_call_id == "call_second"
    assert calls[1].function_name == "get_time"
