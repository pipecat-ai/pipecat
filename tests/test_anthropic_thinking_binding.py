#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for keeping Anthropic thinking blocks valid when a conversation changes."""

import copy
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.adapters.llm.services.anthropic_adapter import (
    AnthropicLLMAdapter,
    AnthropicLLMInvocationParams,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.anthropic.thinking_binding import ThinkingBinding, is_binding_error

BINDING_ERROR = (
    "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a "
    "different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_"
    'behavior` to "drop_block". The `tools` list differs from the one this block was created with.'
)
TAMPERED_ERROR = "messages.1.content.0: Invalid `signature` in `thinking` block"
OTHER_ERROR = "messages: text content blocks must be non-empty"


def _tool(name: str) -> FunctionSchema:
    return FunctionSchema(
        name=name,
        description=f"The {name} tool.",
        properties={"city": {"type": "string"}},
        required=["city"],
    )


def _thought(signature: str, llm: str = "anthropic") -> LLMSpecificMessage:
    return LLMSpecificMessage(
        llm=llm, message={"type": "thought", "text": "", "signature": signature}
    )


def _context() -> LLMContext:
    context = LLMContext(tools=ToolsSchema(standard_tools=[_tool("get_time")]))
    context.add_message({"role": "user", "content": "What time is it in Helsinki?"})
    context.add_message(_thought("sig-1"))
    context.add_message({"role": "assistant", "content": "It's 14:05."})
    context.add_message({"role": "user", "content": "And the weather?"})
    return context


def _add_turn(context: LLMContext, signature: str):
    context.add_message(_thought(signature))
    context.add_message({"role": "assistant", "content": "Sunny."})
    context.add_message({"role": "user", "content": "Thanks."})


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


def _params(system="You are a voice assistant.", tools=None) -> AnthropicLLMInvocationParams:
    return {"system": system, "messages": [], "tools": tools or [{"name": "get_time"}]}


class _Api:
    """Stands in for the Anthropic API: records requests, rejecting the first ones given."""

    def __init__(self, *errors: Exception, during_first: Callable[[], None] | None = None):
        self.requests: list[dict] = []
        self._errors = list(errors)
        self._during_first = during_first

    @property
    def sent(self) -> list[list[str]]:
        return [_signatures(request["messages"]) for request in self.requests]

    async def create(self, api_call, params):
        self.requests.append({**params, "messages": copy.deepcopy(params["messages"])})
        if self._during_first:
            self._during_first()
            self._during_first = None
        if self._errors:
            raise self._errors.pop(0)

        async def no_events():
            return
            yield

        return no_events()


async def _run(service: AnthropicLLMService, context: LLMContext, api: _Api):
    with patch.object(service, "_create_message_stream", api.create):
        await service._process_context(context)


def _enforced() -> ThinkingBinding:
    binding = ThinkingBinding()
    binding.enforced = True
    return binding


def test_binding_changes_only_count_once_enforced():
    binding, context = ThinkingBinding(), LLMContext()
    binding.record(context, _params())
    assert not binding.changed(context, _params(system="You are a weather assistant."))


def test_binding_changes_with_system_prompt_or_tools():
    binding, context = _enforced(), LLMContext()
    assert not binding.changed(context, _params())
    binding.record(context, _params())
    assert not binding.changed(context, _params())
    assert binding.changed(context, _params(system="You are a weather assistant."))
    assert binding.changed(context, _params(tools=[{"name": "get_time"}, {"name": "get_weather"}]))


def test_binding_ignores_cache_markers_and_tool_order():
    binding, context = _enforced(), LLMContext()
    binding.record(context, _params(tools=[{"name": "a"}, {"name": "b"}]))
    system = [{"type": "text", "text": "You are a voice assistant.", "cache_control": {"x": 1}}]
    tools = [{"name": "b", "cache_control": {"x": 1}}, {"name": "a"}]
    assert not binding.changed(context, _params(system=system, tools=tools))


def test_binding_sees_a_tool_parameter_named_cache_control():
    binding, context = _enforced(), LLMContext()
    tool = {"name": "a", "input_schema": {"properties": {"cache_control": {"type": "string"}}}}
    binding.record(context, _params(tools=[tool]))
    changed = copy.deepcopy(tool)
    changed["input_schema"]["properties"]["cache_control"]["type"] = "integer"
    assert binding.changed(context, _params(tools=[changed]))


def test_binding_is_kept_per_context():
    binding, first, second = _enforced(), LLMContext(), LLMContext()
    binding.record(first, _params())
    binding.record(second, _params(tools=[{"name": "get_weather"}]))
    assert not binding.changed(first, _params())
    assert not binding.changed(second, _params(tools=[{"name": "get_weather"}]))


def test_is_binding_error():
    assert is_binding_error(_bad_request(BINDING_ERROR))
    assert not is_binding_error(_bad_request(TAMPERED_ERROR))
    assert not is_binding_error(_bad_request(OTHER_ERROR))


def test_adapter_removes_only_its_own_thoughts():
    adapter = AnthropicLLMAdapter()
    context = _context()
    context.add_message(_thought("google-sig", llm="google"))

    assert adapter.remove_thoughts(context)
    assert not adapter.remove_thoughts(context)
    assert [m for m in context.messages if isinstance(m, LLMSpecificMessage)] == [
        _thought("google-sig", llm="google")
    ]


@pytest.mark.asyncio
async def test_thinking_kept_until_anthropic_rejects_it():
    service, context, api = _make_service(), _context(), _Api()

    await _run(service, context, api)
    context.set_tools(ToolsSchema(standard_tools=[_tool("get_time"), _tool("get_weather")]))
    await _run(service, context, api)

    assert api.sent == [["sig-1"], ["sig-1"]]


@pytest.mark.asyncio
async def test_rejection_retries_without_thinking_and_removes_it():
    service, context, api = _make_service(), _context(), _Api(_bad_request(BINDING_ERROR))

    await _run(service, context, api)
    await _run(service, context, api)

    assert api.sent == [["sig-1"], [], []]
    first, retry = api.requests[0], api.requests[1]
    assert {k: v for k, v in first.items() if k != "messages"} == {
        k: v for k, v in retry.items() if k != "messages"
    }
    assert not any(isinstance(m, LLMSpecificMessage) for m in context.messages)
    assert service.is_usable


@pytest.mark.asyncio
async def test_after_a_rejection_a_tool_change_removes_earlier_thinking_up_front():
    service, context, api = _make_service(), _context(), _Api(_bad_request(BINDING_ERROR))

    await _run(service, context, api)
    _add_turn(context, "sig-2")
    await _run(service, context, api)
    context.set_tools(ToolsSchema(standard_tools=[_tool("get_time"), _tool("get_weather")]))
    await _run(service, context, api)
    _add_turn(context, "sig-3")
    await _run(service, context, api)

    assert api.sent == [["sig-1"], [], ["sig-2"], [], ["sig-3"]]


@pytest.mark.asyncio
async def test_after_a_rejection_a_system_prompt_change_removes_earlier_thinking_up_front():
    service, context, api = _make_service(), _context(), _Api(_bad_request(BINDING_ERROR))

    await _run(service, context, api)
    _add_turn(context, "sig-2")
    service._settings.system_instruction = "You are a weather assistant."
    await _run(service, context, api)

    assert api.sent == [["sig-1"], [], []]


@pytest.mark.asyncio
async def test_retry_leaves_out_messages_added_during_the_request():
    context = _context()

    def user_speaks():
        context.add_message({"role": "user", "content": "Also, is it raining?"})

    service, api = _make_service(), _Api(_bad_request(BINDING_ERROR), during_first=user_speaks)

    await _run(service, context, api)

    first, retry = api.requests[0]["messages"], api.requests[1]["messages"]
    assert len(retry) == len(first)
    assert "raining" not in str(retry)


@pytest.mark.asyncio
async def test_retry_keeps_a_tool_round_valid():
    context = LLMContext(tools=ToolsSchema(standard_tools=[_tool("get_time")]))
    context.add_message({"role": "user", "content": "What time is it in Helsinki?"})
    context.add_message(_thought("sig-1"))
    context.add_message(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": '{"city": "Helsinki"}'},
                }
            ],
        }
    )
    context.add_message({"role": "tool", "tool_call_id": "t1", "content": '{"time": "14:05"}'})
    service, api = _make_service(), _Api(_bad_request(BINDING_ERROR))

    await _run(service, context, api)

    retry = api.requests[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert [b["type"] for b in retry[1]["content"]] == ["tool_use"]
    assert [b["type"] for b in retry[2]["content"]] == ["tool_result"]


@pytest.mark.asyncio
async def test_other_bad_request_is_not_retried():
    service, context, api = _make_service(), _context(), _Api(_bad_request(OTHER_ERROR))

    await _run(service, context, api)

    assert api.sent == [["sig-1"]]
    assert not service.is_usable


@pytest.mark.asyncio
async def test_rejection_is_retried_only_once():
    rejections = (_bad_request(BINDING_ERROR), _bad_request(BINDING_ERROR))
    service, context, api = _make_service(), _context(), _Api(*rejections)

    await _run(service, context, api)

    assert api.sent == [["sig-1"], []]
    assert not service.is_usable


@pytest.mark.asyncio
async def test_another_service_on_the_same_context_does_not_resend_removed_thinking():
    context = _context()
    await _run(_make_service(), context, _Api(_bad_request(BINDING_ERROR)))
    _add_turn(context, "sig-2")

    api = _Api()
    await _run(_make_service(), context, api)

    assert api.sent == [["sig-2"]]


@pytest.mark.asyncio
async def test_run_inference_retries_without_thinking_and_keeps_the_context():
    service, context = _make_service(), _context()
    reply = SimpleNamespace(content=[SimpleNamespace(text="Sunny.")])
    create = AsyncMock(side_effect=[_bad_request(BINDING_ERROR), reply])
    service._client.beta.messages.create = create

    result = await service.run_inference(context, system_instruction="Summarize the call.")

    assert result == "Sunny."
    assert [_signatures(call.kwargs["messages"]) for call in create.call_args_list] == [
        ["sig-1"],
        [],
    ]
    assert _thought("sig-1") in context.messages


@pytest.mark.asyncio
async def test_run_inference_raises_other_bad_requests():
    service, context = _make_service(), _context()
    create = AsyncMock(side_effect=_bad_request(OTHER_ERROR))
    service._client.beta.messages.create = create

    with pytest.raises(anthropic.BadRequestError):
        await service.run_inference(context)

    assert create.await_count == 1
