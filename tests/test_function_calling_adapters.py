#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from openai.types.chat import ChatCompletionToolParam
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.tool_search_tool_param import ToolSearchToolParam

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.adapters.services.inworld_realtime_adapter import InworldRealtimeLLMAdapter
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.adapters.services.open_ai_realtime_adapter import OpenAIRealtimeLLMAdapter
from pipecat.adapters.services.open_ai_responses_adapter import OpenAIResponsesLLMAdapter


class TestFunctionAdapters(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common tools schema for all tests."""
        function_def = FunctionSchema(
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
        self.tools_def = ToolsSchema(standard_tools=[function_def])

    def test_openai_adapter(self):
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
        assert OpenAILLMAdapter().to_provider_tools_format(self.tools_def) == expected

    def test_anthropic_adapter(self):
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
        assert AnthropicLLMAdapter().to_provider_tools_format(self.tools_def) == expected

    def test_gemini_adapter(self):
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
        assert GeminiLLMAdapter().to_provider_tools_format(self.tools_def) == expected

    def test_openai_realtime_adapter(self):
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
        assert OpenAIRealtimeLLMAdapter().to_provider_tools_format(self.tools_def) == expected

    def test_inworld_realtime_adapter(self):
        """Test Inworld Realtime adapter format transformation."""
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
        assert InworldRealtimeLLMAdapter().to_provider_tools_format(self.tools_def) == expected

    def test_gemini_adapter_with_custom_tools(self):
        """Test Gemini adapter format transformation."""
        search_tool = {"google_search": {}}
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
            },
            search_tool,
        ]
        tools_def = self.tools_def
        tools_def.custom_tools = {AdapterType.GEMINI: [search_tool]}
        assert GeminiLLMAdapter().to_provider_tools_format(tools_def) == expected

    def test_openai_adapter_with_custom_tools(self):
        """Test OpenAI adapter appends custom tools."""
        tool_search = {"type": "tool_search"}
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
            ),
            tool_search,
        ]
        tools_def = self.tools_def
        tools_def.custom_tools = {AdapterType.OPENAI: [tool_search]}
        assert OpenAILLMAdapter().to_provider_tools_format(tools_def) == expected

    def test_openai_responses_adapter_with_custom_tools(self):
        """Test OpenAI Responses adapter appends custom tools."""
        tool_search = {"type": "tool_search"}
        expected = [
            FunctionToolParam(
                type="function",
                name="get_weather",
                description="Get the weather in a given location",
                parameters={
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
                strict=None,
            ),
            ToolSearchToolParam(type="tool_search"),
        ]
        tools_def = self.tools_def
        tools_def.custom_tools = {AdapterType.OPENAI: [tool_search]}
        assert OpenAIResponsesLLMAdapter().to_provider_tools_format(tools_def) == expected

    def test_openai_responses_adapter(self):
        """Test OpenAI Responses adapter format transformation."""
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
                "strict": None,
            }
        ]
        assert OpenAIResponsesLLMAdapter().to_provider_tools_format(self.tools_def) == expected

    def test_openai_realtime_adapter_with_custom_tools(self):
        """Test OpenAI Realtime adapter appends custom tools."""
        tool_search = {"type": "tool_search"}
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
            },
            tool_search,
        ]
        tools_def = self.tools_def
        tools_def.custom_tools = {AdapterType.OPENAI: [tool_search]}
        assert OpenAIRealtimeLLMAdapter().to_provider_tools_format(tools_def) == expected

    def test_openai_adapter_ignores_other_adapter_custom_tools(self):
        """Test that OpenAI adapter ignores custom tools for other adapters."""
        expected = OpenAILLMAdapter().to_provider_tools_format(self.tools_def)
        tools_def = self.tools_def
        tools_def.custom_tools = {AdapterType.GEMINI: [{"google_search": {}}]}
        assert OpenAILLMAdapter().to_provider_tools_format(tools_def) == expected

    def test_bedrock_adapter(self):
        """Test AWS Bedrock adapter format transformation."""
        expected = [
            {
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "format": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The temperature unit to use.",
                                },
                                "location": {
                                    "type": "string",
                                    "description": "The city, e.g. San Francisco",
                                },
                            },
                            "required": ["location", "format"],
                        }
                    },
                }
            }
        ]
        assert AWSBedrockLLMAdapter().to_provider_tools_format(self.tools_def) == expected


if __name__ == "__main__":
    unittest.main()
