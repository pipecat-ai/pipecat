#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for leaving out thinking blocks the conversation no longer matches."""

from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.anthropic_adapter import (
    AnthropicLLMAdapter,
    AnthropicLLMInvocationParams,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.anthropic.thinking_binding import StaleThinking, is_binding_error

BINDING_ERROR = (
    "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a "
    "different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_"
    'behavior` to "drop_block". The `tools` list differs from the one this block was created with.'
)


def _tool(name: str) -> FunctionSchema:
    return FunctionSchema(
        name=name,
        description=f"The {name} tool.",
        properties={"city": {"type": "string"}},
        required=["city"],
    )


def _thought(signature: str) -> LLMSpecificMessage:
    return LLMSpecificMessage(
        llm="anthropic", message={"type": "thought", "text": "", "signature": signature}
    )


def _context() -> LLMContext:
    context = LLMContext(tools=ToolsSchema(standard_tools=[_tool("get_time")]))
    context.add_message({"role": "user", "content": "What time is it in Helsinki?"})
    context.add_message(_thought("sig-1"))
    context.add_message({"role": "assistant", "content": "It's 14:05."})
    context.add_message({"role": "user", "content": "And the weather?"})
    return context


def _make_service() -> AnthropicLLMService:
    service = AnthropicLLMService(api_key="test-key")
    service.push_frame = AsyncMock()
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    return service


def _bad_request(message: str) -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    response = httpx.Response(400, request=request, json=body)
    return anthropic.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)


def _signatures(messages: list) -> list[str]:
    return [
        block["signature"]
        for message in messages
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "thinking"
    ]


class _Requests:
    """Stands in for the Anthropic API: records requests, optionally rejecting some."""

    def __init__(self, *errors: Exception):
        self.sent: list[list[str]] = []
        self._errors = list(errors)

    async def create(self, api_call, params):
        self.sent.append(_signatures(params["messages"]))
        if self._errors:
            raise self._errors.pop(0)

        async def no_events():
            return
            yield

        return no_events()


async def _run(service: AnthropicLLMService, context: LLMContext, requests: _Requests):
    with patch.object(service, "_create_message_stream", requests.create):
        await service._process_context(context)


def _params(system="You are a voice assistant.", tools=None) -> AnthropicLLMInvocationParams:
    return {"system": system, "messages": [], "tools": tools or [{"name": "get_time"}]}


def test_binding_changes_with_system_prompt_or_tools():
    stale = StaleThinking()
    assert not stale.binding_changed(_params())
    assert not stale.binding_changed(_params())
    assert stale.binding_changed(_params(system="You are a weather assistant."))
    assert stale.binding_changed(_params(tools=[{"name": "get_time"}, {"name": "get_weather"}]))


def test_binding_ignores_cache_markers_and_tool_order():
    stale = StaleThinking()
    stale.binding_changed(_params(tools=[{"name": "a"}, {"name": "b"}]))
    cached_system = [
        {
            "type": "text",
            "text": "You are a voice assistant.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert not stale.binding_changed(
        _params(system=cached_system, tools=[{"name": "b"}, {"name": "a"}])
    )


def test_add_reports_only_new_signatures():
    stale = StaleThinking()
    assert stale.add({"sig-1"})
    assert not stale.add({"sig-1"})
    assert stale.add({"sig-1", "sig-2"})
    assert stale.signatures == {"sig-1", "sig-2"}


def test_is_binding_error():
    assert is_binding_error(_bad_request(BINDING_ERROR))
    assert not is_binding_error(_bad_request("messages: text content blocks must be non-empty"))


@pytest.mark.asyncio
async def test_adapter_leaves_out_excluded_thoughts():
    adapter = AnthropicLLMAdapter()
    context = _context()
    context.add_message(_thought("sig-2"))
    context.add_message({"role": "assistant", "content": "Sunny."})

    assert adapter.thought_signatures(context) == {"sig-1", "sig-2"}

    params = await adapter.get_llm_invocation_params(
        context, enable_prompt_caching=False, excluded_thought_signatures={"sig-1"}
    )
    assert _signatures(params["messages"]) == ["sig-2"]
    assert [m["role"] for m in params["messages"]] == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_thinking_kept_while_system_prompt_and_tools_are_unchanged():
    service = _make_service()
    context = _context()
    requests = _Requests()

    await _run(service, context, requests)
    await _run(service, context, requests)

    assert requests.sent == [["sig-1"], ["sig-1"]]


@pytest.mark.asyncio
async def test_tool_change_leaves_out_earlier_thinking_from_then_on():
    service = _make_service()
    context = _context()
    requests = _Requests()

    await _run(service, context, requests)
    context.set_tools(ToolsSchema(standard_tools=[_tool("get_time"), _tool("get_weather")]))
    await _run(service, context, requests)
    # Thinking created after the change is bound to the new tools and kept.
    context.add_message(_thought("sig-2"))
    context.add_message({"role": "assistant", "content": "Sunny."})
    context.add_message({"role": "user", "content": "Thanks."})
    await _run(service, context, requests)

    assert requests.sent == [["sig-1"], [], ["sig-2"]]


@pytest.mark.asyncio
async def test_system_prompt_change_leaves_out_earlier_thinking():
    service = _make_service()
    context = _context()
    requests = _Requests()

    service._settings.system_instruction = "You are a voice assistant."
    await _run(service, context, requests)
    service._settings.system_instruction = "You are a weather assistant."
    await _run(service, context, requests)

    assert requests.sent == [["sig-1"], []]


@pytest.mark.asyncio
async def test_tool_order_and_prompt_caching_are_not_changes():
    service = _make_service()
    context = LLMContext(
        messages=_context().get_messages(),
        tools=ToolsSchema(standard_tools=[_tool("get_time"), _tool("get_weather")]),
    )
    requests = _Requests()

    service._settings.system_instruction = "You are a voice assistant."
    await _run(service, context, requests)
    context.set_tools(ToolsSchema(standard_tools=[_tool("get_weather"), _tool("get_time")]))
    service._settings.enable_prompt_caching = True
    await _run(service, context, requests)

    assert requests.sent == [["sig-1"], ["sig-1"]]


@pytest.mark.asyncio
async def test_binding_rejection_retries_without_thinking_and_keeps_it_out():
    service = _make_service()
    context = _context()
    requests = _Requests(_bad_request(BINDING_ERROR))

    await _run(service, context, requests)
    await _run(service, context, requests)

    assert requests.sent == [["sig-1"], [], []]
    assert service.is_usable


@pytest.mark.asyncio
async def test_other_bad_request_is_not_retried():
    service = _make_service()
    context = _context()
    requests = _Requests(_bad_request("messages: text content blocks must be non-empty"))

    await _run(service, context, requests)

    assert requests.sent == [["sig-1"]]
    assert not service.is_usable


@pytest.mark.asyncio
async def test_binding_rejection_is_retried_only_once():
    service = _make_service()
    context = _context()
    requests = _Requests(_bad_request(BINDING_ERROR), _bad_request(BINDING_ERROR))

    await _run(service, context, requests)

    assert requests.sent == [["sig-1"], []]
    assert not service.is_usable
