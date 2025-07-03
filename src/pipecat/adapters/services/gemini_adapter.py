#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini LLM adapter for Pipecat."""

from typing import Any, Dict, List, Union

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema


class GeminiLLMAdapter(BaseLLMAdapter):
    """LLM adapter for Google's Gemini service.

    Provides tool schema conversion functionality to transform standard tool
    definitions into Gemini's specific function-calling format for use with
    Gemini LLM models.
    """

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Convert tool schemas to Gemini's function-calling format.

        Args:
            tools_schema: The tools schema containing standard and custom tool definitions.

        Returns:
            List of tool definitions formatted for Gemini's function-calling API.
            Includes both converted standard tools and any custom Gemini-specific tools.
        """
        functions_schema = tools_schema.standard_tools
        formatted_standard_tools = [
            {"function_declarations": [func.to_default_dict() for func in functions_schema]}
        ]
        custom_gemini_tools = []
        if tools_schema.custom_tools:
            custom_gemini_tools = tools_schema.custom_tools.get(AdapterType.GEMINI, [])

        return formatted_standard_tools + custom_gemini_tools
