#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the OpenAI Responses API adapter.

Tests the conversion from LLMContext messages to Responses API input items, including:

1. Simple user/assistant text messages pass through (with correct role)
2. System role converted to developer role
3. First-message system role triggers a warning
4. Assistant messages with tool_calls produce function_call input items
5. Tool messages produce function_call_output input items
6. Mixed conversations with text + function calls convert correctly
7. Multimodal content conversion (text -> input_text, image_url -> input_image)
8. Tools schema flattening (nested function dict -> flat format)
9. Empty messages list
10. LLMSpecificMessage with llm="openai_responses" passes through
"""

import unittest
from unittest.mock import patch

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_responses_adapter import OpenAIResponsesLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext, LLMStandardMessage


class TestOpenAIResponsesAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIResponsesLLMAdapter()

    def test_simple_user_assistant_messages(self):
        """Simple user/assistant text messages are converted correctly."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 2)
        self.assertEqual(params["input"][0], {"role": "user", "content": "Hello"})
        self.assertEqual(params["input"][1], {"role": "assistant", "content": "Hi there!"})

    def test_system_role_converted_to_developer(self):
        """System role messages are converted to developer role."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["input"][0]["role"], "developer")
        self.assertEqual(params["input"][0]["content"], "You are helpful.")

    def test_first_system_message_triggers_warning(self):
        """First system message triggers a warning about using system_instruction."""
        # Use a fresh adapter so the warning hasn't been emitted yet
        adapter = OpenAIResponsesLLMAdapter()
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.services.open_ai_responses_adapter.logger") as mock_logger:
            adapter.get_llm_invocation_params(context)
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("system_instruction", warning_msg)

    def test_non_initial_system_message_no_warning(self):
        """Non-initial system messages are converted without a warning."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "New instruction"},
        ]
        context = LLMContext(messages=messages)

        adapter = OpenAIResponsesLLMAdapter()
        with patch("pipecat.adapters.services.open_ai_responses_adapter.logger") as mock_logger:
            params = adapter.get_llm_invocation_params(context)
            mock_logger.warning.assert_not_called()

        self.assertEqual(params["input"][1]["role"], "developer")
        self.assertEqual(params["input"][1]["content"], "New instruction")

    def test_first_system_message_warning_fires_only_once(self):
        """The first-system-message warning fires only once per adapter instance."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        adapter = OpenAIResponsesLLMAdapter()
        with patch("pipecat.adapters.services.open_ai_responses_adapter.logger") as mock_logger:
            adapter.get_llm_invocation_params(context)
            adapter.get_llm_invocation_params(context)
            # Warning should have been emitted exactly once, not twice
            mock_logger.warning.assert_called_once()

    def test_assistant_tool_calls_to_function_call(self):
        """Assistant messages with tool_calls produce function_call input items."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}',
                        },
                        "type": "function",
                    }
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 1)
        fc = params["input"][0]
        self.assertEqual(fc["type"], "function_call")
        self.assertEqual(fc["call_id"], "call_123")
        self.assertEqual(fc["name"], "get_weather")
        self.assertEqual(fc["arguments"], '{"location": "SF"}')

    def test_multiple_tool_calls(self):
        """Multiple tool calls in one assistant message produce multiple function_call items."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                        "type": "function",
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "get_restaurant", "arguments": '{"location": "SF"}'},
                        "type": "function",
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 2)
        self.assertEqual(params["input"][0]["name"], "get_weather")
        self.assertEqual(params["input"][1]["name"], "get_restaurant")

    def test_tool_message_to_function_call_output(self):
        """Tool role messages produce function_call_output input items."""
        messages = [
            {
                "role": "tool",
                "content": '{"temperature": "72"}',
                "tool_call_id": "call_123",
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 1)
        fco = params["input"][0]
        self.assertEqual(fco["type"], "function_call_output")
        self.assertEqual(fco["call_id"], "call_123")
        self.assertEqual(fco["output"], '{"temperature": "72"}')

    def test_mixed_conversation(self):
        """Mixed conversation with text + function calls converts correctly."""
        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                        "type": "function",
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temp": "72"}',
                "tool_call_id": "call_abc",
            },
            {"role": "assistant", "content": "It's 72 degrees in SF."},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 4)
        self.assertEqual(params["input"][0]["role"], "user")
        self.assertEqual(params["input"][1]["type"], "function_call")
        self.assertEqual(params["input"][2]["type"], "function_call_output")
        self.assertEqual(params["input"][3]["role"], "assistant")

    def test_multimodal_text_conversion(self):
        """Multimodal text content parts are converted to input_text."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        content = params["input"][0]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[0]["text"], "What's in this image?")

    def test_multimodal_image_conversion(self):
        """Multimodal image_url content parts are converted to input_image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123"},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        content = params["input"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["image_url"], "data:image/jpeg;base64,abc123")
        self.assertEqual(content[1]["detail"], "auto")

    def test_multimodal_image_with_detail(self):
        """Image content parts preserve the detail setting when provided."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png", "detail": "high"},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        content = params["input"][0]["content"]
        self.assertEqual(content[0]["detail"], "high")

    def test_tools_schema_flattening(self):
        """Tools schema with nested function dict is flattened to Responses API format."""
        weather_fn = FunctionSchema(
            name="get_weather",
            description="Get the current weather",
            properties={
                "location": {"type": "string", "description": "The city"},
            },
            required=["location"],
        )
        tools = ToolsSchema(standard_tools=[weather_fn])
        context = LLMContext(tools=tools)
        params = self.adapter.get_llm_invocation_params(context)

        tool_list = params["tools"]
        self.assertEqual(len(tool_list), 1)
        tool = tool_list[0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], "get_weather")
        self.assertEqual(tool["description"], "Get the current weather")
        self.assertIn("properties", tool["parameters"])

    def test_empty_messages(self):
        """Empty messages list produces empty input list."""
        context = LLMContext(messages=[])
        params = self.adapter.get_llm_invocation_params(context)
        self.assertEqual(params["input"], [])

    def test_llm_specific_message_passthrough(self):
        """LLMSpecificMessage with llm='openai_responses' passes through."""
        specific_msg = self.adapter.create_llm_specific_message(
            {"type": "function_call", "call_id": "x", "name": "foo", "arguments": "{}"}
        )
        messages = [
            {"role": "user", "content": "Hello"},
            specific_msg,
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 2)
        self.assertEqual(params["input"][0]["role"], "user")
        self.assertEqual(params["input"][1]["type"], "function_call")

    def test_id_for_llm_specific_messages(self):
        """Adapter identifier is 'openai_responses'."""
        self.assertEqual(self.adapter.id_for_llm_specific_messages, "openai_responses")

    def test_system_instruction_with_messages_sets_instructions(self):
        """When system_instruction is provided and input is non-empty, sets instructions."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")

        self.assertEqual(params["instructions"], "Be helpful.")
        self.assertEqual(len(params["input"]), 1)
        self.assertEqual(params["input"][0]["role"], "user")

    def test_system_instruction_with_empty_input_becomes_developer_message(self):
        """When system_instruction is provided but input is empty, it becomes a developer message."""
        context = LLMContext(messages=[])
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")

        self.assertNotIn("instructions", params)
        self.assertEqual(len(params["input"]), 1)
        self.assertEqual(params["input"][0]["role"], "developer")
        self.assertEqual(params["input"][0]["content"], "Be helpful.")

    def test_no_system_instruction_omits_instructions(self):
        """When no system_instruction is provided, instructions key is absent."""
        context = LLMContext(messages=[{"role": "user", "content": "Hi"}])
        params = self.adapter.get_llm_invocation_params(context)

        self.assertNotIn("instructions", params)


if __name__ == "__main__":
    unittest.main()
