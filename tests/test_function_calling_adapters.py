#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from openai.types.chat import ChatCompletionToolParam

from pipecat.adapters.function_schema import FunctionSchema
from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.adapters.services.open_ai_realtime_adapter import OpenAIRealtimeLLMAdapter


class TestFunctionAdapters(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common function schema for all tests."""
        self.function_def = FunctionSchema(
            name="get_weather",
            description="Get the weather in a given location",
            properties={
                "location": {"type": "string", "description": "The city, e.g. San Francisco"},
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
                },
            },
            required=["location", "format"],
        )

    def test_openai_adapter_single_tool(self):
        """Test OpenAI adapter format transformation."""
        expected = ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get the weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city, e.g. San Francisco",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        )
        assert OpenAILLMAdapter().to_provider_function_format(self.function_def) == expected

    def test_openai_adapter_multiple_tools(self):
        """Test OpenAI adapter format transformation."""
        expected = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city, e.g. San Francisco",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use.",
                            },
                        },
                        "required": ["location", "format"],
                    },
                },
            )
        ]
        assert OpenAILLMAdapter().to_provider_function_format([self.function_def]) == expected

    def test_anthropic_adapter_single_tool(self):
        """Test Anthropic adapter format transformation."""
        expected = {
            "name": "get_weather",
            "description": "Get the weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city, e.g. San Francisco",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use.",
                    },
                },
                "required": ["location", "format"],
            },
        }
        assert AnthropicLLMAdapter().to_provider_function_format(self.function_def) == expected

    def test_anthropic_adapter_multiple_tools(self):
        """Test Anthropic adapter format transformation."""
        expected = [
            {
                "name": "get_weather",
                "description": "Get the weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city, e.g. San Francisco",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use.",
                        },
                    },
                    "required": ["location", "format"],
                },
            }
        ]
        assert AnthropicLLMAdapter().to_provider_function_format([self.function_def]) == expected

    def test_gemini_adapter_single_tool(self):
        """Test Gemini adapter format transformation."""
        expected = {
            "name": "get_weather",
            "description": "Get the weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city, e.g. San Francisco",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use.",
                    },
                },
                "required": ["location", "format"],
            },
        }
        assert GeminiLLMAdapter().to_provider_function_format(self.function_def) == expected

    def test_gemini_adapter_multiple_tools(self):
        """Test Gemini adapter format transformation."""
        expected = [
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city, e.g. San Francisco",
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The temperature unit to use.",
                                },
                            },
                            "required": ["location", "format"],
                        },
                    }
                ]
            }
        ]
        assert GeminiLLMAdapter().to_provider_function_format([self.function_def]) == expected

    def test_openai_realtime_adapter_single_tool(self):
        """Test Anthropic adapter format transformation."""
        expected = {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city, e.g. San Francisco",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use.",
                    },
                },
                "required": ["location", "format"],
            },
        }
        assert OpenAIRealtimeLLMAdapter().to_provider_function_format(self.function_def) == expected

    def test_openai_realtime_adapter_multiple_tools(self):
        """Test Anthropic adapter format transformation."""
        expected = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city, e.g. San Francisco",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use.",
                        },
                    },
                    "required": ["location", "format"],
                },
            }
        ]
        assert OpenAIRealtimeLLMAdapter().to_provider_function_format([self.function_def]) == expected
