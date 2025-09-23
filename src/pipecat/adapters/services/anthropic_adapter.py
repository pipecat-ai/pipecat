#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anthropic LLM adapter for Pipecat."""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

from anthropic import NOT_GIVEN, NotGiven
from anthropic.types.message_param import MessageParam
from anthropic.types.tool_union_param import ToolUnionParam
from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    LLMStandardMessage,
)


class AnthropicLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking Anthropic's LLM API."""

    system: str | NotGiven
    messages: List[MessageParam]
    tools: List[ToolUnionParam]


class AnthropicLLMAdapter(BaseLLMAdapter[AnthropicLLMInvocationParams]):
    """Adapter for converting tool schemas to Anthropic's function-calling format.

    This adapter handles the conversion of Pipecat's standard function schemas
    to the specific format required by Anthropic's Claude models for function calling.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for Anthropic."""
        return "anthropic"

    def get_llm_invocation_params(
        self, context: LLMContext, enable_prompt_caching: bool
    ) -> AnthropicLLMInvocationParams:
        """Get Anthropic-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.
            enable_prompt_caching: Whether prompt caching should be enabled.

        Returns:
            Dictionary of parameters for invoking Anthropic's LLM API.
        """
        messages = self._from_universal_context_messages(self.get_messages(context))
        return {
            "system": messages.system,
            "messages": (
                self._with_cache_control_markers(messages.messages)
                if enable_prompt_caching
                else messages.messages
            ),
            # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": self.from_standard_tools(context.tools) or [],
        }

    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about Anthropic.

        Removes or truncates sensitive data like image content for safe logging.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about Anthropic.
        """
        # Get messages in Anthropic's format
        messages = self._from_universal_context_messages(self.get_messages(context)).messages

        # Sanitize messages for logging
        messages_for_logging = []
        for message in messages:
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "image":
                            item["source"]["data"] = "..."
            messages_for_logging.append(msg)
        return messages_for_logging

    @dataclass
    class ConvertedMessages:
        """Container for Anthropic-formatted messages converted from universal context."""

        messages: List[MessageParam]
        system: str | NotGiven

    def _from_universal_context_messages(
        self, universal_context_messages: List[LLMContextMessage]
    ) -> ConvertedMessages:
        system = NOT_GIVEN
        messages = []

        # first, map messages using self._from_universal_context_message(m)
        try:
            messages = [self._from_universal_context_message(m) for m in universal_context_messages]
        except Exception as e:
            logger.error(f"Error mapping messages: {e}")

        # See if we should pull the system message out of our messages list.
        if messages and messages[0]["role"] == "system":
            if len(messages) == 1:
                # If we have only have a system message in the list, all we can really do
                # without introducing too much magic is change the role to "user".
                messages[0]["role"] = "user"
            else:
                # If we have more than one message, we'll pull the system message out of the
                # list.
                system = messages[0]["content"]
                messages.pop(0)

        # Convert any subsequent "system"-role messages to "user"-role
        # messages, as Anthropic doesn't support system input messages.
        for message in messages:
            if message["role"] == "system":
                message["role"] = "user"

        # Merge consecutive messages with the same role.
        i = 0
        while i < len(messages) - 1:
            current_message = messages[i]
            next_message = messages[i + 1]
            if current_message["role"] == next_message["role"]:
                # Convert content to list of dictionaries if it's a string
                if isinstance(current_message["content"], str):
                    current_message["content"] = [
                        {"type": "text", "text": current_message["content"]}
                    ]
                if isinstance(next_message["content"], str):
                    next_message["content"] = [{"type": "text", "text": next_message["content"]}]
                # Concatenate the content
                current_message["content"].extend(next_message["content"])
                # Remove the next message from the list
                messages.pop(i + 1)
            else:
                i += 1

        # Avoid empty content in messages
        for message in messages:
            if isinstance(message["content"], str) and message["content"] == "":
                message["content"] = "(empty)"
            elif isinstance(message["content"], list) and len(message["content"]) == 0:
                message["content"] = [{"type": "text", "text": "(empty)"}]

        return self.ConvertedMessages(messages=messages, system=system)

    def _from_universal_context_message(self, message: LLMContextMessage) -> MessageParam:
        if isinstance(message, LLMSpecificMessage):
            return copy.deepcopy(message.message)
        return self._from_standard_message(message)

    def _from_standard_message(self, message: LLMStandardMessage) -> MessageParam:
        """Convert standard universal context message to Anthropic format.

        Handles conversion of text content, tool calls, and tool results.
        Empty text content is converted to "(empty)".

        Args:
            message: Message in standard universal context format.

        Returns:
            Message in Anthropic format.

        Examples:
            Input standard format::

                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "123",
                            "function": {"name": "search", "arguments": '{"q": "test"}'}
                        }
                    ]
                }

            Output Anthropic format::

                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "123",
                            "name": "search",
                            "input": {"q": "test"}
                        }
                    ]
                }
        """
        message = copy.deepcopy(message)
        if message["role"] == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                    },
                ],
            }
        if message.get("tool_calls"):
            tc = message["tool_calls"]
            ret = {"role": "assistant", "content": []}
            for tool_call in tc:
                function = tool_call["function"]
                arguments = json.loads(function["arguments"])
                new_tool_use = {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": function["name"],
                    "input": arguments,
                }
                ret["content"].append(new_tool_use)
            return ret
        content = message.get("content")
        if isinstance(content, str):
            # fix empty text
            if content == "":
                content = "(empty)"
        elif isinstance(content, list):
            for item in content:
                # fix empty text
                if item["type"] == "text" and item["text"] == "":
                    item["text"] = "(empty)"
                # handle image_url -> image conversion
                if item["type"] == "image_url":
                    item["type"] = "image"
                    item["source"] = {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": item["image_url"]["url"].split(",")[1],
                    }
                    del item["image_url"]
            # In the case where there's a single image in the list (like what
            # would result from a UserImageRawFrame), ensure that the image
            # comes before text, as recommended by Anthropic docs
            # (https://docs.anthropic.com/en/docs/build-with-claude/vision#example-one-image)
            image_indices = [i for i, item in enumerate(content) if item["type"] == "image"]
            text_indices = [i for i, item in enumerate(content) if item["type"] == "text"]
            if len(image_indices) == 1 and text_indices:
                img_idx = image_indices[0]
                first_txt_idx = text_indices[0]
                if img_idx > first_txt_idx:
                    # Move image before the first text
                    image_item = content.pop(img_idx)
                    content.insert(first_txt_idx, image_item)

        return message

    def _with_cache_control_markers(self, messages: List[MessageParam]) -> List[MessageParam]:
        """Add cache control markers to messages for prompt caching.

        Args:
            messages: List of messages in Anthropic format.

        Returns:
            List of messages with cache control markers added.
        """

        def add_cache_control_marker(message: MessageParam):
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            message["content"][-1]["cache_control"] = {"type": "ephemeral"}

        try:
            # Add cache control markers to the most recent two user messages.
            # - The marker at the most recent user message tells Anthropic to
            #   cache the prompt up to that point.
            # - The marker at the second-most-recent user message tells Anthropic
            #   to look up the cached prompt that goes up to that point (the
            #   point that *was* the last user message the previous turn).
            # If we only added the marker to the last user message, we'd only
            # ever be adding to the cache, never looking up from it.
            # Why user messages? We're assuming that we're primarily running
            # inference as soon as user turns come in. In Anthropic, turns
            # strictly alternate between user and assistant.

            messages_with_markers = copy.deepcopy(messages)

            # Find the most recent two user messages
            user_message_indices = []
            for i in range(len(messages_with_markers) - 1, -1, -1):
                if messages_with_markers[i]["role"] == "user":
                    user_message_indices.append(i)
                    if len(user_message_indices) == 2:
                        break

            # Add cache control markers to the identified user messages
            for index in user_message_indices:
                add_cache_control_marker(messages_with_markers[index])

            return messages_with_markers
        except Exception as e:
            logger.error(f"Error adding cache control marker: {e}")
            return messages_with_markers

    @staticmethod
    def _to_anthropic_function_format(function: FunctionSchema) -> Dict[str, Any]:
        """Convert a single function schema to Anthropic's format.

        Args:
            function: The function schema to convert.

        Returns:
            Dictionary containing the function definition in Anthropic's format.
        """
        return {
            "name": function.name,
            "description": function.description,
            "input_schema": {
                "type": "object",
                "properties": function.properties,
                "required": function.required,
            },
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Convert function schemas to Anthropic's function-calling format.

        Args:
            tools_schema: The tools schema containing functions to convert.

        Returns:
            List of function definitions formatted for Anthropic's API.
        """
        functions_schema = tools_schema.standard_tools
        return [self._to_anthropic_function_format(func) for func in functions_schema]
