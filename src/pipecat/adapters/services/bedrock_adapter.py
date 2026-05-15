#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Bedrock LLM adapter for Pipecat."""

import base64
import copy
import json
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMContextToolChoice,
    LLMSpecificMessage,
    LLMStandardMessage,
)


class AWSBedrockLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking AWS Bedrock's LLM API."""

    system: list[dict[str, Any]] | None  # [{"text": "system message"}]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    tool_choice: LLMContextToolChoice


class AWSBedrockLLMAdapter(BaseLLMAdapter[AWSBedrockLLMInvocationParams]):
    """Adapter for AWS Bedrock LLM integration with Pipecat.

    Provides conversion utilities for transforming Pipecat function schemas
    into AWS Bedrock's expected tool format for function calling capabilities.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for AWS Bedrock."""
        return "aws"

    def get_llm_invocation_params(
        self, context: LLMContext, *, system_instruction: str | None = None
    ) -> AWSBedrockLLMInvocationParams:
        """Get AWS Bedrock-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings
                or ``run_inference``.

        Returns:
            Dictionary of parameters for invoking AWS Bedrock's LLM API.
        """
        converted = self._from_universal_context_messages(
            self.get_messages(context), system_instruction=system_instruction
        )
        effective_system = self._resolve_system_instruction(
            converted.system,
            system_instruction,
            discard_context_system=True,
        )
        return cast(
            AWSBedrockLLMInvocationParams,
            {
                "system": [{"text": effective_system}] if effective_system else None,
                "messages": converted.messages,
                # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
                "tools": self.from_standard_tools(context.tools) or [],
                # To avoid refactoring in AWSBedrockLLMService, we just pass through tool_choice.
                # Eventually (when we don't have to maintain the non-LLMContext code path) we should do
                # the conversion to Bedrock's expected format here rather than in AWSBedrockLLMService.
                "tool_choice": context.tool_choice,
            },
        )

    def get_messages_for_logging(self, context) -> list[dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about AWS Bedrock.

        Removes or truncates sensitive data like image content for safe logging.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about AWS Bedrock.
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
                        if item.get("image"):
                            item["image"]["source"]["bytes"] = "..."
            messages_for_logging.append(msg)
        return messages_for_logging

    @dataclass
    class ConvertedMessages:
        """Container for Bedrock-formatted messages converted from universal context."""

        messages: list[dict[str, Any]]
        system: str | None

    def _from_universal_context_messages(
        self,
        universal_context_messages: list[LLMContextMessage],
        *,
        system_instruction: str | None = None,
    ) -> ConvertedMessages:
        system = None

        # Extract initial system message from universal messages BEFORE conversion,
        # so the helper works with standard message format (not provider-specific).
        remaining = list(universal_context_messages)
        if remaining and not isinstance(remaining[0], LLMSpecificMessage):
            system = self._extract_initial_system(remaining, system_instruction=system_instruction)

        # Convert remaining messages to Bedrock format
        messages = []
        try:
            messages = [self._from_universal_context_message(m) for m in remaining]
        except Exception as e:
            logger.error(f"Error mapping messages: {e}")

        # Convert any subsequent "system"/"developer"-role messages to "user"-role
        # messages, as AWS Bedrock doesn't support system or developer input messages.
        for message in messages:
            if message["role"] in ("system", "developer"):
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

    def _from_universal_context_message(self, message: LLMContextMessage) -> dict[str, Any]:
        if isinstance(message, LLMSpecificMessage):
            return copy.deepcopy(message.message)
        return self._from_standard_message(message)

    def _from_standard_message(self, message: LLMStandardMessage) -> dict[str, Any]:
        """Convert standard format message to AWS Bedrock format.

        Handles conversion of text content, tool calls, and tool results.
        Empty text content is converted to "(empty)".

        Args:
            message: Message in standard format.

        Returns:
            Message in AWS Bedrock format.

        Examples:
            Standard format input::

                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "123",
                            "function": {"name": "search", "arguments": '{"q": "test"}'}
                        }
                    ]
                }

            AWS Bedrock format output::

                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "123",
                                "name": "search",
                                "input": {"q": "test"}
                            }
                        }
                    ]
                }
        """
        # ChatCompletionMessageParam (input) and the dict shape Bedrock expects
        # are different — work with the deepcopied message as a plain dict for
        # the transformations below.
        msg = cast(dict[str, Any], copy.deepcopy(message))
        if msg["role"] == "tool":
            # Try to parse the content as JSON if it looks like JSON
            try:
                if msg["content"].strip().startswith("{") and msg["content"].strip().endswith("}"):
                    content_json = json.loads(msg["content"])
                    tool_result_content = [{"json": content_json}]
                else:
                    tool_result_content = [{"text": msg["content"]}]
            except (json.JSONDecodeError, ValueError, AttributeError):
                tool_result_content = [{"text": msg["content"]}]

            return {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": msg["tool_call_id"],
                            "content": tool_result_content,
                        },
                    },
                ],
            }

        if msg.get("tool_calls"):
            tc = msg["tool_calls"]
            ret: dict[str, Any] = {"role": "assistant", "content": []}
            for tool_call in tc:
                function = tool_call["function"]
                arguments = json.loads(function["arguments"])
                new_tool_use = {
                    "toolUse": {
                        "toolUseId": tool_call["id"],
                        "name": function["name"],
                        "input": arguments,
                    }
                }
                ret["content"].append(new_tool_use)
            return ret

        # Handle text content
        content = msg.get("content")
        if isinstance(content, str):
            if content == "":
                return {"role": msg["role"], "content": [{"text": "(empty)"}]}
            else:
                return {"role": msg["role"], "content": [{"text": content}]}
        elif isinstance(content, list):
            new_content = []
            for item in content:
                # fix empty text
                if item.get("type", "") == "text":
                    text_content = item["text"] if item["text"] != "" else "(empty)"
                    new_content.append({"text": text_content})
                # handle image_url -> image conversion
                if item["type"] == "image_url":
                    if item["image_url"]["url"].startswith("data:"):
                        # Extract format from data URL (format: "data:image/jpeg;base64,...")
                        url = item["image_url"]["url"]
                        mime_type = url.split(":")[1].split(";")[0]
                        # Bedrock expects format like "jpeg", "png" etc., not "image/jpeg"
                        image_format = mime_type.split("/")[1]
                        new_item = {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": base64.b64decode(url.split(",")[1])},
                            }
                        }
                        new_content.append(new_item)
                    else:
                        url = item["image_url"]["url"]
                        logger.warning(f"Unsupported 'image_url': {url}")

            # In the case where there's a single image in the list (like what
            # would result from a UserImageRawFrame), ensure that the image
            # comes before text
            image_indices = [i for i, item in enumerate(new_content) if "image" in item]
            text_indices = [i for i, item in enumerate(new_content) if "text" in item]
            if len(image_indices) == 1 and text_indices:
                img_idx = image_indices[0]
                first_txt_idx = text_indices[0]
                if img_idx > first_txt_idx:
                    # Move image before the first text
                    image_item = new_content.pop(img_idx)
                new_content.insert(first_txt_idx, image_item)
            return {"role": msg["role"], "content": new_content}

        return msg

    @staticmethod
    def _to_bedrock_function_format(function: FunctionSchema) -> dict[str, Any]:
        """Convert a function schema to Bedrock's tool format.

        Args:
            function: The function schema to convert.

        Returns:
            Dictionary formatted for Bedrock's tool specification.
        """
        return {
            "toolSpec": {
                "name": function.name,
                "description": function.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": function.properties,
                        "required": function.required,
                    },
                },
            }
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[dict[str, Any]]:
        """Convert function schemas to Bedrock's function-calling format.

        Args:
            tools_schema: The tools schema containing functions to convert.

        Returns:
            List of Bedrock formatted function call definitions.
        """
        functions_schema = tools_schema.standard_tools
        return [self._to_bedrock_function_format(func) for func in functions_schema]
