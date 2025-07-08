#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime LLM adapter for Pipecat."""

from typing import Any, Dict, List, Union

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema


class OpenAIRealtimeLLMAdapter(BaseLLMAdapter):
    """LLM adapter for OpenAI Realtime API function calling.

    Converts Pipecat's tool schemas into the specific format required by
    OpenAI's Realtime API for function calling capabilities.
    """

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
        return [self._to_openai_realtime_function_format(func) for func in functions_schema]
