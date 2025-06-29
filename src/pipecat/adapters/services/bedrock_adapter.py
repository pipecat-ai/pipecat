#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Bedrock LLM adapter for Pipecat."""

from typing import Any, Dict, List

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema


class AWSBedrockLLMAdapter(BaseLLMAdapter):
    """Adapter for AWS Bedrock LLM integration with Pipecat.

    Provides conversion utilities for transforming Pipecat function schemas
    into AWS Bedrock's expected tool format for function calling capabilities.
    """

    @staticmethod
    def _to_bedrock_function_format(function: FunctionSchema) -> Dict[str, Any]:
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

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Convert function schemas to Bedrock's function-calling format.

        Args:
            tools_schema: The tools schema containing functions to convert.

        Returns:
            List of Bedrock formatted function call definitions.
        """
        functions_schema = tools_schema.standard_tools
        return [self._to_bedrock_function_format(func) for func in functions_schema]
