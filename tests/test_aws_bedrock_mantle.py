#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for using OpenAILLMService against AWS Bedrock Mantle's OpenAI-compatible API.

AWS Bedrock Mantle exposes an OpenAI-compatible Chat Completions API, so
Pipecat talks to it through the existing OpenAILLMService by pointing
base_url at the Mantle endpoint and passing the Bedrock API key as api_key.
These tests confirm that path end-to-end without any AWS credentials or
network access: client configuration, request construction, streaming, and
tool calling all go through OpenAILLMService/AsyncOpenAI unchanged.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import AsyncOpenAI

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.tests.utils import run_test

MANTLE_BASE_URL = "https://bedrock-mantle.us-east-1.api.aws/v1"


def _text_chunk(content: str):
    """Build a stream chunk carrying a text delta, as Mantle's OpenAI-compatible API would send."""
    return SimpleNamespace(
        usage=None,
        model="test-model",
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))],
    )


def _tool_call_chunk(index: int, tool_call_id: str | None, name: str | None, arguments: str):
    """Build a stream chunk carrying a tool-call delta."""
    return SimpleNamespace(
        usage=None,
        model="test-model",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=index,
                            id=tool_call_id,
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                )
            )
        ],
    )


def test_mantle_client_configuration():
    """OpenAILLMService builds an AsyncOpenAI client pointed at the Mantle endpoint.

    No AWS access key/secret is required: the Bedrock API key is passed
    straight through as the OpenAI client's api_key.
    """
    service = OpenAILLMService(
        api_key="test-key",
        base_url=MANTLE_BASE_URL,
        settings=OpenAILLMService.Settings(model="test-model"),
    )

    assert isinstance(service._client, AsyncOpenAI)
    assert service._client.api_key == "test-key"
    assert str(service._client.base_url) == f"{MANTLE_BASE_URL}/"


@pytest.mark.asyncio
async def test_mantle_chat_completions_request():
    """The request sent to chat.completions.create matches what Mantle expects."""
    service = OpenAILLMService(
        api_key="test-key",
        base_url=MANTLE_BASE_URL,
        settings=OpenAILLMService.Settings(model="test-model"),
    )
    service._client = AsyncMock()
    service._client.chat.completions.create = AsyncMock(return_value=_FakeStream([]))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()

    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
    await service._process_context(context)

    call_kwargs = service._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["stream"] is True
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


class _FakeStream:
    """Stands in for the AsyncOpenAI chat completion stream."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_mantle_streaming_produces_llm_text_frames():
    """A Mantle-style ChatCompletionChunk stream produces LLMTextFrames downstream.

    This confirms Mantle can reuse the existing OpenAI streaming loop without
    a provider-specific adapter.
    """
    service = OpenAILLMService(
        api_key="test-key",
        base_url=MANTLE_BASE_URL,
        settings=OpenAILLMService.Settings(model="test-model"),
    )
    service._client = AsyncMock()
    service.get_chat_completions = AsyncMock(
        return_value=_FakeStream([_text_chunk("Hello"), _text_chunk(" there")])
    )
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()

    context = LLMContext(messages=[{"role": "user", "content": "Hi"}])

    down_frames, _ = await run_test(
        service,
        frames_to_send=[LLMContextFrame(context=context)],
        expected_down_frames=[
            LLMServiceMetadataFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ],
    )

    texts = [frame.text for frame in down_frames if isinstance(frame, LLMTextFrame)]
    assert texts == ["Hello", " there"]


@pytest.mark.asyncio
async def test_mantle_tool_calling_request():
    """A Pipecat tool definition reaches chat.completions.create in OpenAI's format."""
    get_weather = FunctionSchema(
        name="get_weather",
        description="Get weather information",
        properties={"city": {"type": "string"}},
        required=["city"],
    )

    service = OpenAILLMService(
        api_key="test-key",
        base_url=MANTLE_BASE_URL,
        settings=OpenAILLMService.Settings(model="test-model"),
    )
    service._client = AsyncMock()
    service._client.chat.completions.create = AsyncMock(return_value=_FakeStream([]))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()

    context = LLMContext(
        messages=[{"role": "user", "content": "What's the weather in Boston?"}],
        tools=[get_weather],
    )
    await service._process_context(context)

    call_kwargs = service._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_mantle_tool_call_streaming_triggers_function_call():
    """Streamed tool-call deltas are coalesced into a function call, as with OpenAI."""
    service = OpenAILLMService(
        api_key="test-key",
        base_url=MANTLE_BASE_URL,
        settings=OpenAILLMService.Settings(model="test-model"),
    )
    service._client = AsyncMock()
    service.get_chat_completions = AsyncMock(
        return_value=_FakeStream(
            [
                _tool_call_chunk(0, "call_123", "get_weather", '{"city": '),
                _tool_call_chunk(0, None, None, '"Boston"}'),
            ]
        )
    )
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.run_function_calls = AsyncMock()

    context = LLMContext(messages=[{"role": "user", "content": "What's the weather in Boston?"}])
    await service._process_context(context)

    service.run_function_calls.assert_called_once()
    function_calls = service.run_function_calls.call_args.args[0]
    assert len(function_calls) == 1
    assert function_calls[0].function_name == "get_weather"
    assert function_calls[0].arguments == {"city": "Boston"}
    assert function_calls[0].tool_call_id == "call_123"


def test_mantle_default_headers_are_forwarded_to_the_client():
    """Optional Mantle-specific headers reach the underlying AsyncOpenAI client."""
    with patch.object(
        OpenAILLMService,
        "create_client",
        return_value=AsyncMock(),
    ) as create_mock:
        OpenAILLMService(
            api_key="test-key",
            base_url=MANTLE_BASE_URL,
            default_headers={"X-Mantle-Trace": "enabled"},
            settings=OpenAILLMService.Settings(model="test-model"),
        )

    kwargs = create_mock.call_args.kwargs
    assert kwargs["base_url"] == MANTLE_BASE_URL
    assert kwargs["api_key"] == "test-key"
    assert kwargs["default_headers"] == {"X-Mantle-Trace": "enabled"}
