#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic LLM adapter for Pipecat."""

import json
from typing import Any, Dict, List, TypedDict

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext


class AWSNovaSonicLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking AWS Nova Sonic LLM API.

    This is a placeholder until support for universal LLMContext machinery is added for AWS Nova Sonic.
    """

    pass


class AWSNovaSonicLLMAdapter(BaseLLMAdapter[AWSNovaSonicLLMInvocationParams]):
    """Adapter for AWS Nova Sonic language models.

    Converts Pipecat's standard function schemas into AWS Nova Sonic's
    specific function-calling format, enabling tool use with Nova Sonic models.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for AWS Nova Sonic."""
        raise NotImplementedError("Universal LLMContext is not yet supported for AWS Nova Sonic.")

    def get_llm_invocation_params(self, context: LLMContext) -> AWSNovaSonicLLMInvocationParams:
        """Get AWS Nova Sonic-specific LLM invocation parameters from a universal LLM context.

        This is a placeholder until support for universal LLMContext machinery is added for AWS Nova Sonic.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for invoking AWS Nova Sonic's LLM API.
        """
        raise NotImplementedError("Universal LLMContext is not yet supported for AWS Nova Sonic.")

    def get_messages_for_logging(self, context) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about AWS Nova Sonic.

        Removes or truncates sensitive data like image content for safe logging.

        This is a placeholder until support for universal LLMContext machinery is added for AWS Nova Sonic.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about AWS Nova Sonic.
        """
        raise NotImplementedError("Universal LLMContext is not yet supported for AWS Nova Sonic.")

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
