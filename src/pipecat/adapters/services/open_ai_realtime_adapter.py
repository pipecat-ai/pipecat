#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
from typing import Any, Dict, List, Union

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.function_schema import FunctionSchema


class OpenAIRealtimeLLMAdapter(BaseLLMAdapter):
    def _to_openai_realtime_function_format(self, function: FunctionSchema) -> Dict[str, Any]:
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

    def to_provider_function_format(
        self, functions_schema: Union[FunctionSchema, List[FunctionSchema]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Converts one or multiple function schemas to Openai Realtime function-calling format.

        :return: Openai Realtime formatted function call definition.
        """

        if isinstance(functions_schema, list):
            # Handling list of FunctionSchema
            return [self._to_openai_realtime_function_format(func) for func in functions_schema]
        else:
            # Handling single FunctionSchema
            return self._to_openai_realtime_function_format(functions_schema)
