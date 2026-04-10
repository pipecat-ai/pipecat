#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM adapter for Pipecat."""

from typing import Any, Dict, List, Optional, TypedDict

from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMContextToolChoice,
    LLMSpecificMessage,
    NotGiven,
)


class OpenAILLMInvocationParams(TypedDict):
    """Context-based parameters for invoking OpenAI ChatCompletion API."""

    messages: List[ChatCompletionMessageParam]
    tools: List[ChatCompletionToolParam] | OpenAINotGiven
    tool_choice: ChatCompletionToolChoiceOptionParam | OpenAINotGiven


class OpenAILLMAdapter(BaseLLMAdapter[OpenAILLMInvocationParams]):
    """OpenAI-specific adapter for Pipecat.

    Handles:

    - Extracting parameters for OpenAI's ChatCompletion API from a universal
      LLM context
    - Converting Pipecat's standardized tools schema to OpenAI's function-calling format.
    - Extracting and sanitizing messages from the LLM context for logging about OpenAI.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for OpenAI."""
        return "openai"

    def get_llm_invocation_params(
        self,
        context: LLMContext,
        *,
        system_instruction: Optional[str] = None,
        convert_developer_to_user: bool,
    ) -> OpenAILLMInvocationParams:
        """Get OpenAI-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings
                or ``run_inference``. If provided, prepended as a system message.
            convert_developer_to_user: If True, convert "developer"-role messages
                to "user"-role messages. Used by OpenAI-compatible services that
                don't support the "developer" role.

        Returns:
            Dictionary of parameters for OpenAI's ChatCompletion API.
        """
        messages = self._from_universal_context_messages(
            self.get_messages(context), convert_developer_to_user=convert_developer_to_user
        )

        if system_instruction:
            # Detect initial system message for warning purposes (don't extract)
            initial_content = (
                messages[0].get("content", "")
                if messages and messages[0].get("role") == "system"
                else None
            )
            self._resolve_system_instruction(
                initial_content,
                system_instruction,
                discard_context_system=False,
            )
            messages = [{"role": "system", "content": system_instruction}] + messages

        return {
            "messages": messages,
            # NOTE; LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": self.from_standard_tools(context.tools),
            "tool_choice": context.tool_choice,
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[ChatCompletionToolParam]:
        """Convert function schemas to OpenAI's function-calling format.

        Args:
            tools_schema: The Pipecat tools schema to convert.

        Returns:
            List of OpenAI formatted function call definitions ready for use
            with ChatCompletion API.
        """
        functions_schema = tools_schema.standard_tools
        formatted_standard_tools = [
            ChatCompletionToolParam(type="function", function=func.to_default_dict())
            for func in functions_schema
        ]
        custom_openai_tools = []
        if tools_schema.custom_tools:
            custom_openai_tools = tools_schema.custom_tools.get(AdapterType.OPENAI, [])
        return formatted_standard_tools + custom_openai_tools

    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about OpenAI.

        Binary data (images, audio) is replaced with short placeholders.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about OpenAI.
        """
        return self.get_messages(context, elide_large_values=True)

    def _from_universal_context_messages(
        self,
        messages: List[LLMContextMessage],
        *,
        convert_developer_to_user: bool,
    ) -> List[ChatCompletionMessageParam]:
        result = []
        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                # Extract the actual message content from LLMSpecificMessage
                result.append(message.message)
            else:
                # Standard message, pass through unchanged
                result.append(message)

        if convert_developer_to_user:
            for msg in result:
                if msg.get("role") == "developer":
                    msg["role"] = "user"

        return result

    def _from_standard_tool_choice(
        self, tool_choice: LLMContextToolChoice | NotGiven
    ) -> ChatCompletionToolChoiceOptionParam | OpenAINotGiven:
        # Just a pass-through: tool_choice is already the right type
        return tool_choice
