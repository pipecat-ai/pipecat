#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anthropic LLM adapter for Pipecat."""

from typing import Any, Dict, List, TypedDict

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext


class AnthropicLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking Anthropic's LLM API.

    This is a placeholder until support for universal LLMContext machinery is added for Anthropic.
    """

    pass


class AnthropicLLMAdapter(BaseLLMAdapter[AnthropicLLMInvocationParams]):
    """Adapter for converting tool schemas to Anthropic's function-calling format.

    This adapter handles the conversion of Pipecat's standard function schemas
    to the specific format required by Anthropic's Claude models for function calling.
    """

    def get_llm_invocation_params(self, context: LLMContext) -> AnthropicLLMInvocationParams:
        """Get Anthropic-specific LLM invocation parameters from a universal LLM context.

        This is a placeholder until support for universal LLMContext machinery is added for Anthropic.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for invoking Anthropic's LLM API.
        """
        raise NotImplementedError("Universal LLMContext is not yet supported for Anthropic.")

    def get_messages_for_logging(self, context) -> List[dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about Anthropic.

        Removes or truncates sensitive data like image content for safe logging.

        This is a placeholder until support for universal LLMContext machinery is added for Anthropic.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about Anthropic.
        """
        raise NotImplementedError("Universal LLMContext is not yet supported for Anthropic.")

    @staticmethod
    def _to_anthropic_function_format(function: FunctionSchema) -> Dict[str, Any]:
        """Convert a single function schema to Anthropic's format.

        Args:
            function: The function schema to convert.

        Returns:
            Dictionary containing the function definition in Anthropic's format.
        """
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
        """Convert function schemas to Anthropic's function-calling format.

        Args:
            tools_schema: The tools schema containing functions to convert.

        Returns:
            List of function definitions formatted for Anthropic's API.
        """
        functions_schema = tools_schema.standard_tools
        return [self._to_anthropic_function_format(func) for func in functions_schema]
