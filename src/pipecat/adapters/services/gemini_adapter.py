#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List, Union

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema


class GeminiLLMAdapter(BaseLLMAdapter):
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Converts function schemas to Gemini's function-calling format.

        :return: Gemini formatted function call definition.
        """

        functions_schema = tools_schema.standard_tools
        formatted_standard_tools = [
            {"function_declarations": [func.to_default_dict() for func in functions_schema]}
        ]
        custom_gemini_tools = []
        if tools_schema.custom_tools:
            custom_gemini_tools = tools_schema.custom_tools.get(AdapterType.GEMINI, [])

        return formatted_standard_tools + custom_gemini_tools
