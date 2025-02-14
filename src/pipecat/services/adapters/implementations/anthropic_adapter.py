#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List, Union

from pipecat.services.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.services.adapters.function_schema import FunctionSchema


class AnthropicLLMAdapter(BaseLLMAdapter):
    def _to_anthropic_function_format(self, function: FunctionSchema) -> Dict[str, Any]:
        return {
            "name": function.name,
            "description": function.description,
            "input_schema": {
                "type": "object",
                "properties": function.properties,
                "required": function.required,
            },
        }

    def to_provider_function_format(
        self, functions_schema: Union[FunctionSchema, List[FunctionSchema]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Converts one or multiple function schemas to Anthropic's function-calling format.

        :return: Anthropic formatted function call definition.
        """

        if isinstance(functions_schema, list):
            # Handling list of FunctionSchema
            return [self._to_anthropic_function_format(func) for func in functions_schema]
        else:
            # Handling single FunctionSchema
            return self._to_anthropic_function_format(functions_schema)
