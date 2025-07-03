#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic LLM adapter for Pipecat."""

import json
from typing import Any, Dict, List

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema


class AWSNovaSonicLLMAdapter(BaseLLMAdapter):
    """Adapter for AWS Nova Sonic language models.

    Converts Pipecat's standard function schemas into AWS Nova Sonic's
    specific function-calling format, enabling tool use with Nova Sonic models.
    """

    @staticmethod
    def _to_aws_nova_sonic_function_format(function: FunctionSchema) -> Dict[str, Any]:
        """Convert a function schema to AWS Nova Sonic format.

        Args:
            function: The function schema to convert.

        Returns:
            Dictionary in AWS Nova Sonic function format with toolSpec structure.
        """
        return {
            "toolSpec": {
                "name": function.name,
                "description": function.description,
                "inputSchema": {
                    "json": json.dumps(
                        {
                            "type": "object",
                            "properties": function.properties,
                            "required": function.required,
                        }
                    )
                },
            }
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Convert tools schema to AWS Nova Sonic function-calling format.

        Args:
            tools_schema: The tools schema containing function definitions to convert.

        Returns:
            List of dictionaries in AWS Nova Sonic function format.
        """
        functions_schema = tools_schema.standard_tools
        return [self._to_aws_nova_sonic_function_format(func) for func in functions_schema]
