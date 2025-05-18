#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema


class AnthropicLLMAdapter(BaseLLMAdapter):
    @staticmethod
    def _to_anthropic_function_format(function: FunctionSchema) -> Dict[str, Any]:
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
        """Converts function schemas to Anthropic's function-calling format.

        :return: Anthropic formatted function call definition.
        """

        functions_schema = tools_schema.standard_tools
        return [self._to_anthropic_function_format(func) for func in functions_schema]
