#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM adapter for Pipecat."""

import copy
import json
from typing import Any, List

from openai.types.chat import ChatCompletionToolParam

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext


class OpenAILLMAdapter(BaseLLMAdapter):
    """Adapter for converting tool schemas to OpenAI's format.

    Provides conversion utilities for transforming Pipecat's standard tool
    schemas into the format expected by OpenAI's ChatCompletion API for
    function calling capabilities.
    """

    def get_llm_invocation_params(self, context: LLMContext) -> dict[str, Any]:
        """Get OpenAI-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for OpenAI's chat completion API.
        """
        return {
            "messages": context.messages,
            # TODO: doesn't seem right that we may or may not need to convert tools here; they should already be guaranteed to exist in a universal format in the LLMContext, right?
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

    def get_messages_for_logging(self, context: LLMContext) -> List[dict[str, Any]]:
        """Get messages from the LLM context in a format ready for logging about OpenAI.

        Removes or truncates sensitive data like image content for safe logging.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about OpenAI.
        """
        msgs = []
        for message in context.messages:
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
        return json.dumps(msgs, ensure_ascii=False)
