#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Responses API adapter for Pipecat."""

from typing import Any, Dict, List, Optional, TypedDict

from openai._types import NotGiven as OpenAINotGiven
from openai.types.responses import FunctionToolParam, ResponseInputItemParam, ToolParam

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
)


class OpenAIResponsesLLMInvocationParams(TypedDict, total=False):
    """Context-based parameters for invoking OpenAI Responses API."""

    input: List[ResponseInputItemParam]
    tools: List[ToolParam] | OpenAINotGiven
    instructions: str


class OpenAIResponsesLLMAdapter(BaseLLMAdapter[OpenAIResponsesLLMInvocationParams]):
    """OpenAI Responses API adapter for Pipecat.

    Handles:

    - Converting LLMContext messages to Responses API input items
    - Converting Pipecat's standardized tools schema to Responses API function tool format
    - Extracting and sanitizing messages from the LLM context for logging
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances."""
        return "openai_responses"

    def get_llm_invocation_params(
        self,
        context: LLMContext,
        *,
        system_instruction: Optional[str] = None,
    ) -> OpenAIResponsesLLMInvocationParams:
        """Get Responses API invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings.

        Returns:
            Dictionary of parameters for the Responses API.
        """
        messages = self.get_messages(context)

        # Check for conflict: system_instruction + initial system message
        if system_instruction and messages:
            first_msg = messages[0] if not isinstance(messages[0], LLMSpecificMessage) else None
            if first_msg and first_msg.get("role") == "system":
                self._resolve_system_instruction(
                    first_msg.get("content", ""),
                    system_instruction,
                    discard_context_system=False,
                )

        input_items = self._convert_messages_to_input(messages)

        params: OpenAIResponsesLLMInvocationParams = {
            "input": input_items,
            "tools": self.from_standard_tools(context.tools),
        }

        if system_instruction:
            # Compatibility: The Responses API requires at least one input
            # message when instructions are provided. Contexts that worked with
            # OpenAILLMService (system_instruction + empty messages) need the
            # instructions converted to an initial developer message.
            #
            # NOTE: The service layer (OpenAIResponsesLLMService) internally
            # manages `previous_response_id` for incremental context delivery
            # over WebSocket. This runs post-adapter — the adapter always
            # produces the full input list and the service determines what
            # subset to send. This empty-input fallback is therefore only
            # relevant for one-shot or initial calls.
            #
            # If we added support for user-provided explicit
            # `previous_response_id` and/or `conversation_id` (overriding
            # internal management), we'd need to revisit this logic, as it'd
            # be legit to provide instructions without input items. Note that
            # over HTTP, `previous_response_id` requires `store=True` (30-day
            # OpenAI-side storage), which is why the HTTP variant doesn't use
            # it. The WebSocket variant avoids this via a connection-local
            # in-memory cache — see the class docstrings in llm.py.
            if not input_items:
                params["input"] = [{"role": "developer", "content": system_instruction}]
            else:
                params["instructions"] = system_instruction

        return params

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[ToolParam]:
        """Convert function schemas to Responses API function tool format.

        Args:
            tools_schema: The Pipecat tools schema to convert.

        Returns:
            List of Responses API function tool definitions.
        """
        functions_schema = tools_schema.standard_tools
        result = []
        for func in functions_schema:
            d = func.to_default_dict()
            tool: FunctionToolParam = {
                "type": "function",
                "name": d["name"],
                "parameters": d.get("parameters", {}),
                "strict": d.get("strict", None),
            }
            if "description" in d:
                tool["description"] = d["description"]
            result.append(tool)
        custom_openai_tools = []
        if tools_schema.custom_tools:
            custom_openai_tools = tools_schema.custom_tools.get(AdapterType.OPENAI, [])
        return result + custom_openai_tools

    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """Get messages from context in a format ready for logging.

        Binary data (images, audio) is replaced with short placeholders.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging.
        """
        return self.get_messages(context, truncate_large_values=True)

    def _convert_messages_to_input(
        self, messages: List[LLMContextMessage]
    ) -> List[ResponseInputItemParam]:
        """Convert LLMContext messages to Responses API input items.

        Args:
            messages: Messages from the LLMContext.

        Returns:
            List of Responses API input items.
        """
        result: List[ResponseInputItemParam] = []

        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                result.append(message.message)
                continue

            role = message.get("role")

            if role in ("system", "developer"):
                content = message.get("content", "")
                if isinstance(content, list):
                    content = self._convert_multimodal_content(content)
                result.append({"role": "developer", "content": content})

            elif role == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    content = self._convert_multimodal_content(content)
                result.append({"role": "user", "content": content})

            elif role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        result.append(
                            {
                                "type": "function_call",
                                "call_id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "arguments": func.get("arguments", ""),
                            }
                        )
                else:
                    content = message.get("content", "")
                    if isinstance(content, list):
                        content = self._convert_multimodal_content(content)
                    result.append({"role": "assistant", "content": content})

            elif role == "tool":
                content = message.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id", ""),
                        "output": content,
                    }
                )

        return result

    def _convert_multimodal_content(self, content: list) -> list:
        """Convert multimodal content parts to Responses API format.

        Args:
            content: List of content parts from the LLMContext message.

        Returns:
            List of content parts in Responses API format.
        """
        result = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                result.append({"type": "input_text", "text": part.get("text", "")})
            elif part_type == "image_url":
                image_url_obj = part.get("image_url", {})
                result.append(
                    {
                        "type": "input_image",
                        "image_url": image_url_obj.get("url", ""),
                        "detail": image_url_obj.get("detail", "auto"),
                    }
                )
            else:
                # Pass through other types as-is. Note: "input_audio" is not
                # yet supported by the Responses API (coming soon per OpenAI
                # docs) but the LLMContext format already matches the expected
                # shape, so it should work once support is enabled.
                result.append(part)
        return result
