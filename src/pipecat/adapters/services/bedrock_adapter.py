#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Bedrock LLM adapter for Pipecat."""

from typing import Any, Dict, List, TypedDict

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext


class AWSBedrockLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking AWS Bedrock's LLM API.

    This is a placeholder until support for universal LLMContext machinery is added for Bedrock.
    """

    pass


class AWSBedrockLLMAdapter(BaseLLMAdapter[AWSBedrockLLMInvocationParams]):
    """Adapter for AWS Bedrock LLM integration with Pipecat.

    Provides conversion utilities for transforming Pipecat function schemas
    into AWS Bedrock's expected tool format for function calling capabilities.
    """

    def get_llm_invocation_params(self, context: LLMContext) -> AWSBedrockLLMInvocationParams:
        """Get AWS Bedrock-specific LLM invocation parameters from a universal LLM context.

        This is a placeholder until support for universal LLMContext machinery is added for Bedrock.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for invoking AWS Bedrock's LLM API.
        """
        raise NotImplementedError("Universal LLMContext is not yet supported for AWS Bedrock.")

    def get_messages_for_logging(self, context) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about AWS Bedrock.

        Removes or truncates sensitive data like image content for safe logging.

        This is a placeholder until support for universal LLMContext machinery is added for Bedrock.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about AWS Bedrock.
        """
        raise NotImplementedError("Universal LLMContext is not yet supported for AWS Bedrock.")

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
