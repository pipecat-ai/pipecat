#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM adapter for Pipecat."""

from typing import List

from openai.types.chat import ChatCompletionToolParam

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import ToolsSchema


class OpenAILLMAdapter(BaseLLMAdapter):
    """Adapter for converting tool schemas to OpenAI's format.

    Provides conversion utilities for transforming Pipecat's standard tool
    schemas into the format expected by OpenAI's ChatCompletion API for
    function calling capabilities.
    """

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
