#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM adapter for Pipecat."""

import copy
import json
from typing import Any, Dict, List, TypedDict

from openai._types import NOT_GIVEN as OPEN_AI_NOT_GIVEN
from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMContextToolChoice,
    NotGiven,
)


class OpenAILLMInvocationParams(TypedDict):
    """Context-based parameters for invoking OpenAI ChatCompletion API."""

    messages: List[ChatCompletionMessageParam]
    tools: List[ChatCompletionToolParam] | OpenAINotGiven
    tool_choice: ChatCompletionToolChoiceOptionParam | OpenAINotGiven


class OpenAILLMAdapter(BaseLLMAdapter[OpenAILLMInvocationParams]):
    """OpenAI-specific adapter for Pipecat.

    Handles:

    - Extracting parameters for OpenAI's ChatCompletion API from a universal
      LLM context
    - Converting Pipecat's standardized tools schema to OpenAI's function-calling format.
    - Extracting and sanitizing messages from the LLM context for logging about OpenAI.
    """

    def get_llm_invocation_params(self, context: LLMContext) -> OpenAILLMInvocationParams:
        """Get OpenAI-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for OpenAI's ChatCompletion API.
        """
        return {
            "messages": self._from_universal_context_messages(self._get_messages(context)),
            # NOTE; LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": self.from_standard_tools(context.tools),
            "tool_choice": context.tool_choice,
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[ChatCompletionToolParam]:
        """Convert function schemas to OpenAI's function-calling format.

        Args:
            tools_schema: The Pipecat tools schema to convert.

        Returns:
            List of OpenAI formatted function call definitions ready for use
            with ChatCompletion API.
        """
        functions_schema = tools_schema.standard_tools
        return [
            ChatCompletionToolParam(type="function", function=func.to_default_dict())
            for func in functions_schema
        ]

    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about OpenAI.

        Removes or truncates sensitive data like image content for safe logging.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about OpenAI.
        """
        msgs = []
        for message in self._get_messages(context):
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "image_url":
                            if item["image_url"]["url"].startswith("data:image/"):
                                item["image_url"]["url"] = "data:image/..."
            if "mime_type" in msg and msg["mime_type"].startswith("image/"):
                msg["data"] = "..."
            msgs.append(msg)
        return msgs

    def _get_messages(self, context: LLMContext) -> List[LLMContextMessage]:
        return context.get_messages("openai")

    def _from_universal_context_messages(
        self, messages: List[LLMContextMessage]
    ) -> List[ChatCompletionMessageParam]:
        # Just a pass-through: messages are already the right type
        return messages

    def _from_standard_tool_choice(
        self, tool_choice: LLMContextToolChoice | NotGiven
    ) -> ChatCompletionToolChoiceOptionParam | OpenAINotGiven:
        # Just a pass-through: tool_choice is already the right type
        return tool_choice
