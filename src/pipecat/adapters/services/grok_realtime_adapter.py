#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok Realtime LLM adapter for Pipecat.

Converts Pipecat's tool schemas and context into the format required by
Grok's Voice Agent API.
"""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.services.grok.realtime import events


class GrokRealtimeLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking Grok Realtime API.

    Attributes:
        system_instruction: System prompt/instructions for the session.
        messages: List of conversation items formatted for Grok Realtime.
        tools: List of tool definitions (function, web_search, x_search, file_search).
    """

    system_instruction: Optional[str]
    messages: List[events.ConversationItem]
    tools: List[Dict[str, Any]]


class GrokRealtimeLLMAdapter(BaseLLMAdapter):
    """LLM adapter for Grok Voice Agent API.

    Converts Pipecat's universal context and tool schemas into the specific
    format required by Grok's Voice Agent Realtime API.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for Grok Realtime."""
        return "grok-realtime"

    def get_llm_invocation_params(self, context: LLMContext) -> GrokRealtimeLLMInvocationParams:
        """Get Grok Realtime-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for invoking Grok's Voice Agent API.
        """
        messages = self._from_universal_context_messages(self.get_messages(context))
        return {
            "system_instruction": messages.system_instruction,
            "messages": messages.messages,
            "tools": self.from_standard_tools(context.tools) or [],
        }

    def get_messages_for_logging(self, context) -> List[Dict[str, Any]]:
        """Get messages from context in a format safe for logging.

        Removes or truncates sensitive data like audio content.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages with sensitive data redacted.
        """
        msgs = []
        for message in self.get_messages(context):
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "input_audio":
                            item["audio"] = "..."
                        if item.get("type") == "audio":
                            item["audio"] = "..."
            msgs.append(msg)
        return msgs

    @dataclass
    class ConvertedMessages:
        """Container for Grok-formatted messages converted from universal context."""

        messages: List[events.ConversationItem]
        system_instruction: Optional[str] = None

    def _from_universal_context_messages(
        self, universal_context_messages: List[LLMContextMessage]
    ) -> ConvertedMessages:
        """Convert universal context messages to Grok Realtime format.

        Similar to OpenAI Realtime, we pack conversation history into a single
        user message since the realtime API doesn't support loading long histories.

        Args:
            universal_context_messages: List of messages in universal format.

        Returns:
            ConvertedMessages with Grok-formatted messages and system instruction.
        """
        if not universal_context_messages:
            return self.ConvertedMessages(messages=[])

        messages = copy.deepcopy(universal_context_messages)
        system_instruction = None

        # Extract system message as session instructions
        if messages[0].get("role") == "system":
            system = messages.pop(0)
            content = system.get("content")
            if isinstance(content, str):
                system_instruction = content
            elif isinstance(content, list):
                system_instruction = content[0].get("text")
            if not messages:
                return self.ConvertedMessages(messages=[], system_instruction=system_instruction)

        # Single user message can be sent normally
        if len(messages) == 1 and messages[0].get("role") == "user":
            return self.ConvertedMessages(
                messages=[self._from_universal_context_message(messages[0])],
                system_instruction=system_instruction,
            )

        # Pack multiple messages into a single user message
        intro_text = """
        This is a previously saved conversation. Please treat this conversation history as a
        starting point for the current conversation."""

        trailing_text = """
        This is the end of the previously saved conversation. Please continue the conversation
        from here. If the last message is a user instruction or question, act on that instruction
        or answer the question. If the last message is an assistant response, simply say that you
        are ready to continue the conversation."""

        return self.ConvertedMessages(
            messages=[
                events.ConversationItem(
                    role="user",
                    type="message",
                    content=[
                        events.ItemContent(
                            type="input_text",
                            text="\n\n".join(
                                [
                                    intro_text,
                                    json.dumps(messages, indent=2),
                                    trailing_text,
                                ]
                            ),
                        )
                    ],
                )
            ],
            system_instruction=system_instruction,
        )

    def _from_universal_context_message(
        self, message: LLMContextMessage
    ) -> events.ConversationItem:
        """Convert a single universal context message to Grok format.

        Args:
            message: Message in universal format.

        Returns:
            ConversationItem formatted for Grok Realtime API.
        """
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                text_content = ""
                for c in content:
                    if c.get("type") == "text":
                        text_content += " " + c.get("text")
                    else:
                        logger.error(
                            f"Unhandled content type in context message: {c.get('type')} - {message}"
                        )
                content = text_content.strip()
            return events.ConversationItem(
                role="user",
                type="message",
                content=[events.ItemContent(type="input_text", text=content)],
            )

        if message.get("role") == "assistant" and message.get("tool_calls"):
            tc = message.get("tool_calls")[0]
            return events.ConversationItem(
                type="function_call",
                call_id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )

        logger.error(f"Unhandled message type in _from_universal_context_message: {message}")

    @staticmethod
    def _to_grok_function_format(function: FunctionSchema) -> Dict[str, Any]:
        """Convert a function schema to Grok Realtime function format.

        Args:
            function: The function schema to convert.

        Returns:
            Dictionary in Grok Realtime function format.
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

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Convert tool schemas to Grok Realtime format.

        Supports both standard function tools and Grok-specific tools
        (web_search, x_search, file_search).

        Args:
            tools_schema: The tools schema containing functions to convert.

        Returns:
            List of tool definitions in Grok Realtime format.
        """
        # Convert standard function tools
        functions_schema = tools_schema.standard_tools
        standard_tools = [self._to_grok_function_format(func) for func in functions_schema]

        # Support shimmed custom tools for backward compatibility
        shimmed_tools = []
        if tools_schema.custom_tools:
            shimmed_tools = tools_schema.custom_tools.get(AdapterType.SHIM, [])

        return standard_tools + shimmed_tools
