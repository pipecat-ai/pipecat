#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime LLM adapter for Pipecat."""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.services.openai.realtime import events


class OpenAIRealtimeLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking OpenAI Realtime API.

    This is a placeholder until support for universal LLMContext machinery is added for OpenAI Realtime.
    """

    system_instruction: Optional[str]
    messages: List[events.ConversationItem]
    tools: List[Dict[str, Any]]


class OpenAIRealtimeLLMAdapter(BaseLLMAdapter):
    """LLM adapter for OpenAI Realtime API function calling.

    Converts Pipecat's tool schemas into the specific format required by
    OpenAI's Realtime API for function calling capabilities.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for OpenAI Realtime."""
        return "openai-realtime"

    def get_llm_invocation_params(self, context: LLMContext) -> OpenAIRealtimeLLMInvocationParams:
        """Get OpenAI Realtime-specific LLM invocation parameters from a universal LLM context.

        This is a placeholder until support for universal LLMContext machinery is added for OpenAI Realtime.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for invoking OpenAI Realtime's API.
        """
        messages = self._from_universal_context_messages(self.get_messages(context))
        return {
            "system_instruction": messages.system_instruction,
            "messages": messages.messages,
            # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": self.from_standard_tools(context.tools) or [],
        }

    def get_messages_for_logging(self, context) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about OpenAI Realtime.

        Removes or truncates sensitive data like image content for safe logging.

        This is a placeholder until support for universal LLMContext machinery is added for OpenAI Realtime.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about OpenAI Realtime.
        """
        # NOTE: this is the same as in OpenAIAdapter, as that's what it was
        # prior to a refactor. Worth noting that for OpenAI Realtime
        # specifically, not everything handled here is necessarily supported
        # (or supported yet).
        msgs = []
        for message in self.get_messages(context):
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "image_url":
                            if item["image_url"]["url"].startswith("data:image/"):
                                item["image_url"]["url"] = "data:image/..."
                        if item["type"] == "input_audio":
                            item["input_audio"]["data"] = "..."
            if "mime_type" in msg and msg["mime_type"].startswith("image/"):
                msg["data"] = "..."
            msgs.append(msg)
        return msgs

    @dataclass
    class ConvertedMessages:
        """Container for OpenAI-formatted messages converted from universal context."""

        messages: List[events.ConversationItem]
        system_instruction: Optional[str] = None

    def _from_universal_context_messages(
        self, universal_context_messages: List[LLMContextMessage]
    ) -> ConvertedMessages:
        # We can't load a long conversation history into the openai realtime api yet. (The API/model
        # forgets that it can do audio, if you do a series of `conversation.item.create` calls.) So
        # our general strategy until this is fixed is just to put everything into a first "user"
        # message as a single input.

        if not universal_context_messages:
            return self.ConvertedMessages(messages=[])

        messages = copy.deepcopy(universal_context_messages)
        system_instruction = None

        # If we have a "system" message as our first message, let's pull that out into session
        # "instructions"
        if messages[0].get("role") == "system":
            system = messages.pop(0)
            content = system.get("content")
            if isinstance(content, str):
                system_instruction = content
            elif isinstance(content, list):
                system_instruction = content[0].get("text")
            if not messages:
                return self.ConvertedMessages(messages=[], system_instruction=system_instruction)

        # If we have just a single "user" item, we can just send it normally
        if len(messages) == 1 and messages[0].get("role") == "user":
            return self.ConvertedMessages(
                messages=[self._from_universal_context_message(messages[0])],
                system_instruction=system_instruction,
            )

        # Otherwise, let's pack everything into a single "user" message with a bit of
        # explanation for the LLM
        intro_text = """
        This is a previously saved conversation. Please treat this conversation history as a
        starting point for the current conversation."""

        trailing_text = """
        This is the end of the previously saved conversation. Please continue the conversation
        from here. If the last message is a user instruction or question, act on that instruction
        or answer the question. If the last message is an assistant response, simple say that you
        are ready to continue the conversation."""

        return self.ConvertedMessages(
            messages=[
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "\n\n".join(
                                [intro_text, json.dumps(messages, indent=2), trailing_text]
                            ),
                        }
                    ],
                }
            ],
            system_instruction=system_instruction,
        )

    def _from_universal_context_message(
        self, message: LLMContextMessage
    ) -> events.ConversationItem:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(message.get("content"), list):
                content = ""
                for c in message.get("content"):
                    if c.get("type") == "text":
                        content += " " + c.get("text")
                    else:
                        logger.error(
                            f"Unhandled content type in context message: {c.get('type')} - {message}"
                        )
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
    def _to_openai_realtime_function_format(function: FunctionSchema) -> Dict[str, Any]:
        """Convert a function schema to OpenAI Realtime format.

        Args:
            function: The function schema to convert.

        Returns:
            Dictionary in OpenAI Realtime function format.
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
        """Convert tool schemas to OpenAI Realtime function-calling format.

        Args:
            tools_schema: The tools schema containing functions to convert.

        Returns:
            List of function definitions in OpenAI Realtime format.
        """
        functions_schema = tools_schema.standard_tools
        standard_tools = [
            self._to_openai_realtime_function_format(func) for func in functions_schema
        ]

        # For backward compatibility, OpenAI Realtime can still be used with
        # tools in dict format, even though it always uses `LLMContext` under
        # the hood (via `LLMContext.from_openai_context()`).
        # To support this behavior, we use "shimmed" custom tools here.
        # (We maintain this backward compatibility because users aren't
        # *knowingly* opting into the new `LLMContext`.)
        shimmed_tools = []
        if tools_schema.custom_tools:
            shimmed_tools = tools_schema.custom_tools.get(AdapterType.SHIM, [])

        return standard_tools + shimmed_tools
