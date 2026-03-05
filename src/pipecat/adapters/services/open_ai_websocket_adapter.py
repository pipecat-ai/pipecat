#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI WebSocket LLM adapter for the Responses API."""

import copy
from typing import Any, Dict, List, Optional, TypedDict

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
)


class OpenAIWebSocketLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking OpenAI Responses API via WebSocket.

    Parameters:
        system_instruction: Optional system instruction extracted from system messages.
        input: List of input items in Responses API format.
        tools: List of tool definitions in Responses API format.
    """

    system_instruction: Optional[str]
    input: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]


class OpenAIWebSocketLLMAdapter(BaseLLMAdapter[OpenAIWebSocketLLMInvocationParams]):
    """LLM adapter for OpenAI Responses API via WebSocket.

    Converts Pipecat's universal LLM context messages into the input format
    required by the OpenAI Responses API (``wss://api.openai.com/v1/responses``).

    Message format mapping:

    - System messages are extracted as ``system_instruction`` (not sent as input items).
    - User messages become ``{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}``.
    - Assistant messages become ``{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}``.
    - Tool calls become ``{"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}``.
    - Tool results become ``{"type": "function_call_output", "call_id": "...", "output": "..."}``.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for OpenAI WebSocket."""
        return "openai-websocket"

    def get_llm_invocation_params(
        self, context: LLMContext, **kwargs
    ) -> OpenAIWebSocketLLMInvocationParams:
        """Get OpenAI Responses API parameters from a universal LLM context.

        Extracts system instructions from system messages and converts all other
        messages into the Responses API input format.

        Args:
            context: The LLM context containing messages, tools, etc.
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary with ``system_instruction``, ``input``, and ``tools`` keys.
        """
        converted = self._convert_messages(self.get_messages(context))
        return {
            "system_instruction": converted["system_instruction"],
            "input": converted["input"],
            "tools": self.from_standard_tools(context.tools) or [],
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Convert tool schemas to OpenAI Responses API function format.

        Args:
            tools_schema: The tools schema containing functions to convert.

        Returns:
            List of function definitions in Responses API format.
        """
        return [self._to_function_tool(func) for func in tools_schema.standard_tools]

    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format safe for logging.

        Truncates binary data such as inline images and audio.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages with sensitive data truncated.
        """
        msgs = []
        for message in self.get_messages(context):
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:image/"):
                                item["image_url"]["url"] = "data:image/..."
                        if item.get("type") == "input_audio":
                            item["input_audio"]["data"] = "..."
            if "mime_type" in msg and msg["mime_type"].startswith("image/"):
                msg["data"] = "..."
            msgs.append(msg)
        return msgs

    def _convert_messages(self, messages: List[LLMContextMessage]) -> Dict[str, Any]:
        """Convert universal context messages to Responses API input format.

        Extracts system instructions and converts remaining messages to input items.

        Args:
            messages: List of universal context messages.

        Returns:
            Dictionary with ``system_instruction`` and ``input`` keys.
        """
        system_instruction: Optional[str] = None
        input_items: List[Dict[str, Any]] = []

        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                # Pass through LLM-specific messages directly as input items
                input_items.append(message.message)
                continue

            role = message.get("role")

            if role == "system":
                system_instruction = self._extract_text_content(message)
                continue

            if role == "user":
                input_items.append(self._convert_user_message(message))
            elif role == "assistant":
                if message.get("tool_calls"):
                    input_items.extend(self._convert_tool_calls(message))
                else:
                    input_items.append(self._convert_assistant_message(message))
            elif role == "tool":
                input_items.append(self._convert_tool_result(message))

        return {
            "system_instruction": system_instruction,
            "input": input_items,
        }

    @staticmethod
    def _extract_text_content(message: LLMContextMessage) -> str:
        """Extract text content from a message.

        Args:
            message: A context message with string or list content.

        Returns:
            The extracted text string.
        """
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(parts) if parts else ""
        return ""

    @staticmethod
    def _convert_user_message(message: LLMContextMessage) -> Dict[str, Any]:
        """Convert a user message to Responses API input format.

        Handles text, image, and multipart content.

        Args:
            message: A user role context message.

        Returns:
            Responses API message item with appropriate content parts.
        """
        content = message.get("content")
        if isinstance(content, str):
            return {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            }
        elif isinstance(content, list):
            converted_content = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    converted_content.append({"type": "input_text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    converted_content.append(
                        {
                            "type": "input_image",
                            "image_url": item.get("image_url", {}).get("url", ""),
                        }
                    )
            if not converted_content:
                converted_content = [{"type": "input_text", "text": ""}]
            return {
                "type": "message",
                "role": "user",
                "content": converted_content,
            }

        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": ""}],
        }

    @staticmethod
    def _convert_assistant_message(message: LLMContextMessage) -> Dict[str, Any]:
        """Convert an assistant message to Responses API input format.

        Args:
            message: An assistant role context message.

        Returns:
            Responses API message item with ``output_text`` content.
        """
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            text = " ".join(text_parts)
        else:
            text = ""

        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }

    @staticmethod
    def _convert_tool_calls(message: LLMContextMessage) -> List[Dict[str, Any]]:
        """Convert an assistant message with tool calls to Responses API format.

        Each tool call becomes a separate ``function_call`` input item.

        Args:
            message: An assistant message containing tool_calls.

        Returns:
            List of ``function_call`` input items.
        """
        items = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            items.append(
                {
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                }
            )
        return items

    @staticmethod
    def _convert_tool_result(message: LLMContextMessage) -> Dict[str, Any]:
        """Convert a tool result message to Responses API format.

        Args:
            message: A tool role context message.

        Returns:
            A ``function_call_output`` input item.
        """
        return {
            "type": "function_call_output",
            "call_id": message.get("tool_call_id", ""),
            "output": message.get("content", ""),
        }

    @staticmethod
    def _to_function_tool(function: FunctionSchema) -> Dict[str, Any]:
        """Convert a function schema to Responses API tool format.

        Args:
            function: The function schema to convert.

        Returns:
            Dictionary in Responses API function tool format.
        """
        return {
            "type": "function",
            "name": function.name,
            "description": function.description,
            "parameters": {
                "type": "object",
                "properties": function.properties,
                "required": function.required,
            },
        }
