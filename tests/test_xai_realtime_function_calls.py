#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for function calls in GrokRealtimeLLMService."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService


def _arguments_done(call_id: str, name: str) -> SimpleNamespace:
    """The fields a ``response.function_call_arguments.done`` handler reads."""
    return SimpleNamespace(call_id=call_id, name=name, arguments="{}")


@pytest.mark.asyncio
async def test_the_calls_of_one_response_run_as_one_group():
    """xAI: "Do not send response.create until all function call outputs have been
    submitted." The calls of a response arrive one at a time; one group makes the
    aggregator run the LLM once, after the last output is in, not once per output."""
    service = GrokRealtimeLLMService(api_key="test-key")
    service._context = LLMContext()
    service.run_function_calls = AsyncMock()  # type: ignore[method-assign]
    service._current_response_id = "resp_1"
    for call_id in ("call_units", "call_faq"):
        service._pending_function_calls[call_id] = SimpleNamespace(call_id=call_id)

    await service._handle_evt_function_call_arguments_done(_arguments_done("call_units", "units"))
    await service._handle_evt_function_call_arguments_done(_arguments_done("call_faq", "faq"))

    calls = service.run_function_calls.call_args_list
    assert [call.args[0][0].tool_call_id for call in calls] == ["call_units", "call_faq"]
    assert [call.kwargs["group_id"] for call in calls] == ["resp_1", "resp_1"]
