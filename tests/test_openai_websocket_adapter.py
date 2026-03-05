#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the OpenAI WebSocket LLM adapter."""

import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_websocket_adapter import (
    OpenAIWebSocketLLMAdapter,
)
from pipecat.processors.aggregators.llm_context import LLMContext


@pytest.fixture
def adapter():
    return OpenAIWebSocketLLMAdapter()


class TestIdForLLMSpecificMessages:
    def test_returns_correct_id(self, adapter):
        assert adapter.id_for_llm_specific_messages == "openai-websocket"


class TestSimpleUserMessage:
    def test_string_content(self, adapter):
        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        params = adapter.get_llm_invocation_params(context)

        assert params["system_instruction"] is None
        assert len(params["input"]) == 1
        item = params["input"][0]
        assert item["type"] == "message"
        assert item["role"] == "user"
        assert item["content"] == [{"type": "input_text", "text": "Hello"}]

    def test_list_content(self, adapter):
        context = LLMContext(
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hello world"}]}]
        )
        params = adapter.get_llm_invocation_params(context)

        assert len(params["input"]) == 1
        item = params["input"][0]
        assert item["content"] == [{"type": "input_text", "text": "Hello world"}]

    def test_multipart_text_content(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part one"},
                        {"type": "text", "text": "Part two"},
                    ],
                }
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        item = params["input"][0]
        assert item["content"] == [
            {"type": "input_text", "text": "Part one"},
            {"type": "input_text", "text": "Part two"},
        ]

    def test_multipart_text_and_image(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        item = params["input"][0]
        assert len(item["content"]) == 2
        assert item["content"][0] == {"type": "input_text", "text": "Describe this"}
        assert item["content"][1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc",
        }


class TestSystemMessageExtraction:
    def test_system_string_content(self, adapter):
        context = LLMContext(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"},
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        assert params["system_instruction"] == "You are a helpful assistant."
        assert len(params["input"]) == 1
        assert params["input"][0]["role"] == "user"

    def test_system_list_content(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be concise."}],
                },
                {"role": "user", "content": "Hi"},
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        assert params["system_instruction"] == "Be concise."

    def test_system_only(self, adapter):
        context = LLMContext(messages=[{"role": "system", "content": "You are helpful."}])
        params = adapter.get_llm_invocation_params(context)

        assert params["system_instruction"] == "You are helpful."
        assert params["input"] == []


class TestAssistantMessages:
    def test_simple_assistant_message(self, adapter):
        context = LLMContext(messages=[{"role": "assistant", "content": "I can help you."}])
        params = adapter.get_llm_invocation_params(context)

        assert len(params["input"]) == 1
        item = params["input"][0]
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert item["content"] == [{"type": "output_text", "text": "I can help you."}]

    def test_assistant_list_content(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Sure thing."}],
                }
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        item = params["input"][0]
        assert item["content"] == [{"type": "output_text", "text": "Sure thing."}]


class TestToolCalls:
    def test_single_tool_call(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                }
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        assert len(params["input"]) == 1
        item = params["input"][0]
        assert item["type"] == "function_call"
        assert item["call_id"] == "call_123"
        assert item["name"] == "get_weather"
        assert item["arguments"] == '{"location": "NYC"}'

    def test_multiple_tool_calls(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "func_a",
                                "arguments": "{}",
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "func_b",
                                "arguments": '{"x": 1}',
                            },
                        },
                    ],
                }
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        assert len(params["input"]) == 2
        assert params["input"][0]["call_id"] == "call_1"
        assert params["input"][1]["call_id"] == "call_2"


class TestToolResults:
    def test_tool_result(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": '{"temp": 72}',
                }
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        assert len(params["input"]) == 1
        item = params["input"][0]
        assert item["type"] == "function_call_output"
        assert item["call_id"] == "call_123"
        assert item["output"] == '{"temp": 72}'


class TestEmptyMessages:
    def test_empty_context(self, adapter):
        context = LLMContext(messages=[])
        params = adapter.get_llm_invocation_params(context)

        assert params["system_instruction"] is None
        assert params["input"] == []
        assert params["tools"] == []

    def test_no_messages(self, adapter):
        context = LLMContext()
        params = adapter.get_llm_invocation_params(context)

        assert params["system_instruction"] is None
        assert params["input"] == []


class TestToolsFormat:
    def test_single_function(self, adapter):
        func = FunctionSchema(
            name="get_weather",
            description="Get the weather",
            properties={
                "location": {"type": "string", "description": "City name"},
            },
            required=["location"],
        )
        tools_schema = ToolsSchema(standard_tools=[func])
        result = adapter.to_provider_tools_format(tools_schema)

        assert len(result) == 1
        assert result[0] == {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        }

    def test_multiple_functions(self, adapter):
        func1 = FunctionSchema(name="func_a", description="A", properties={}, required=[])
        func2 = FunctionSchema(
            name="func_b", description="B", properties={"x": {"type": "int"}}, required=["x"]
        )
        tools_schema = ToolsSchema(standard_tools=[func1, func2])
        result = adapter.to_provider_tools_format(tools_schema)

        assert len(result) == 2
        assert result[0]["name"] == "func_a"
        assert result[1]["name"] == "func_b"


class TestLogging:
    def test_truncates_image_data(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc123..."},
                        }
                    ],
                }
            ]
        )
        msgs = adapter.get_messages_for_logging(context)

        assert msgs[0]["content"][0]["image_url"]["url"] == "data:image/..."

    def test_truncates_audio_data(self, adapter):
        context = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "base64audiodata..."},
                        }
                    ],
                }
            ]
        )
        msgs = adapter.get_messages_for_logging(context)

        assert msgs[0]["content"][0]["input_audio"]["data"] == "..."

    def test_preserves_text_content(self, adapter):
        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        msgs = adapter.get_messages_for_logging(context)

        assert msgs[0]["content"] == "Hello"


class TestFullConversation:
    def test_multi_turn_with_tools(self, adapter):
        """Test a full conversation flow with system, user, assistant, tool call, and tool result."""
        context = LLMContext(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What's the weather in NYC?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "content": '{"temp": 72, "condition": "sunny"}',
                },
                {"role": "assistant", "content": "It's 72F and sunny in NYC."},
                {"role": "user", "content": "Thanks!"},
            ]
        )
        params = adapter.get_llm_invocation_params(context)

        assert params["system_instruction"] == "You are helpful."
        assert len(params["input"]) == 5

        # User message
        assert params["input"][0]["type"] == "message"
        assert params["input"][0]["role"] == "user"

        # Tool call
        assert params["input"][1]["type"] == "function_call"
        assert params["input"][1]["name"] == "get_weather"

        # Tool result
        assert params["input"][2]["type"] == "function_call_output"
        assert params["input"][2]["call_id"] == "call_weather"

        # Assistant message
        assert params["input"][3]["type"] == "message"
        assert params["input"][3]["role"] == "assistant"

        # Final user message
        assert params["input"][4]["type"] == "message"
        assert params["input"][4]["role"] == "user"


class TestLLMSpecificMessages:
    def test_passthrough(self, adapter):
        """LLM-specific messages should be passed through directly."""
        specific_msg = adapter.create_llm_specific_message(
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        )
        context = LLMContext(messages=[specific_msg])
        params = adapter.get_llm_invocation_params(context)

        assert len(params["input"]) == 1
        assert params["input"][0]["type"] == "message"
        assert params["input"][0]["role"] == "user"
