#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Responses API adapter for Pipecat."""

import copy
from collections.abc import Mapping
from typing import Any, Required, TypedDict, cast

from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai._types import NotGiven as OpenAINotGiven
from openai.types.responses import FunctionToolParam, ResponseInputItemParam, ToolParam
from openai.types.responses.response_create_params import ToolChoice as OpenAIResponsesToolChoice

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.adapters.services.open_ai_adapter import openai_from_llm_context_tools
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMContextToolChoice,
    LLMSpecificMessage,
    NotGiven,
)
from pipecat.utils.types import is_given


class OpenAIResponsesLLMInvocationParams(TypedDict, total=False):
    """Context-based parameters for invoking OpenAI Responses API."""

    # `input` and `tools` are always populated by `get_llm_invocation_params`;
    # `instructions` and `tool_choice` are only set when present.
    input: Required[list[ResponseInputItemParam]]
    tools: Required[list[ToolParam] | OpenAINotGiven]
    instructions: str
    tool_choice: OpenAIResponsesToolChoice


def _flatten_named_tool_reference(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a Chat Completions reference to a named tool for the Responses API.

    Chat Completions nests the name in an object keyed by the tool's type
    (``{"type": "function", "function": {"name": ...}}``); the Responses API
    carries it on the reference itself (``{"type": "function", "name": ...}``).
    References that name nothing to unwrap — hosted tools such as
    ``{"type": "image_generation"}`` — are shared between the two APIs and are
    returned unchanged.

    Args:
        tool: A reference to a tool, in Chat Completions shape.

    Returns:
        The reference in Responses API shape.
    """
    tool_type = tool.get("type")
    nested = tool.get(tool_type) if isinstance(tool_type, str) else None
    if isinstance(nested, dict) and "name" in nested:
        return {"type": tool_type, "name": nested["name"]}
    return dict(tool)


def openai_responses_from_llm_context_tool_choice(
    tool_choice: LLMContextToolChoice | NotGiven,
) -> OpenAIResponsesToolChoice | OpenAINotGiven:
    """Reinterpret an LLMContext ``tool_choice`` as the Responses API's type.

    ``LLMContextToolChoice`` is aliased to Chat Completions' tool choice type,
    whose object forms nest their payload one level deeper than the Responses
    API's: a choice naming a tool arrives as
    ``{"type": "function", "function": {"name": ...}}``, and one restricting the
    model to a subset wraps its ``mode`` and ``tools`` in an ``"allowed_tools"``
    object. Both are flattened here, along with the tool references inside a
    subset. The ``"none"`` / ``"auto"`` / ``"required"`` literals are shared
    between the two APIs and pass through unchanged, as is the "not provided"
    sentinel (translated to the SDK's own, the way
    :func:`~pipecat.adapters.services.open_ai_adapter.openai_from_llm_context_tool_choice`
    does for Chat Completions).

    Args:
        tool_choice: A context's tool choice, or its "not provided" sentinel.

    Returns:
        The tool choice reshaped for the Responses API, or the SDK's "not
        provided" sentinel.
    """
    if not is_given(tool_choice):
        return OPENAI_NOT_GIVEN
    if not isinstance(tool_choice, dict):
        return cast("OpenAIResponsesToolChoice", tool_choice)
    if tool_choice.get("type") == "allowed_tools":
        # A choice already in the Responses API's shape carries `mode` and
        # `tools` itself, with nothing nested to lift.
        allowed = tool_choice.get("allowed_tools") or tool_choice
        return cast(
            "OpenAIResponsesToolChoice",
            {
                **allowed,
                "type": "allowed_tools",
                "tools": [_flatten_named_tool_reference(t) for t in allowed.get("tools", [])],
            },
        )
    return cast("OpenAIResponsesToolChoice", _flatten_named_tool_reference(tool_choice))


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
        system_instruction: str | None = None,
    ) -> OpenAIResponsesLLMInvocationParams:
        """Get Responses API invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings.

        Returns:
            Dictionary of parameters for the Responses API.
        """
        messages = self.get_messages(context)

        if messages:
            first_msg = messages[0] if not isinstance(messages[0], LLMSpecificMessage) else None
            if first_msg and first_msg.get("role") == "system":
                self._warn_context_system_message()
                # Check for conflict: system_instruction + initial system message.
                # `content` is `str | Iterable[...]`; we only forward it for
                # warning purposes. Coerce non-strings to None.
                first_content = first_msg.get("content", "")
                if system_instruction:
                    self._resolve_system_instruction(
                        first_content if isinstance(first_content, str) else None,
                        system_instruction,
                        discard_context_system=False,
                    )

        input_items = self._convert_messages_to_input(messages)

        params: OpenAIResponsesLLMInvocationParams = {
            "input": input_items,
            # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": openai_from_llm_context_tools(self.from_standard_tools(context.tools)),
        }

        resolved_tool_choice = openai_responses_from_llm_context_tool_choice(context.tool_choice)
        if not isinstance(resolved_tool_choice, OpenAINotGiven):
            params["tool_choice"] = resolved_tool_choice

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

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[ToolParam]:
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

    def get_messages_for_logging(self, context: LLMContext) -> list[dict[str, Any]]:
        """Get messages from context in a format ready for logging.

        Binary data (images, audio) is replaced with short placeholders, and
        reasoning messages' encrypted payloads are elided.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging.
        """
        # Sanitize messages for logging
        messages_for_logging: list[dict[str, Any]] = []
        for message in self.get_messages(context, truncate_large_values=True):
            if isinstance(message, LLMSpecificMessage):
                # Responses-specific messages are reasoning items (see
                # _BaseOpenAIResponsesLLMService._append_reasoning_message).
                # Elide the encrypted payload, which is opaque noise in logs.
                msg: dict[str, Any] = copy.deepcopy(message.message)
                if isinstance(msg, dict) and msg.get("encrypted_content"):
                    msg["encrypted_content"] = "..."
                messages_for_logging.append(msg)
            else:
                messages_for_logging.append(cast(dict[str, Any], message))
        return messages_for_logging

    def _convert_messages_to_input(
        self, messages: list[LLMContextMessage]
    ) -> list[ResponseInputItemParam]:
        """Convert LLMContext messages to Responses API input items.

        Args:
            messages: Messages from the LLMContext.

        Returns:
            List of Responses API input items.
        """
        result: list[ResponseInputItemParam] = []

        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                result.append(self._from_specific_message(message))
                continue

            role = message.get("role")

            if role in ("system", "developer"):
                content = message.get("content", "")
                if isinstance(content, list):
                    content = self._convert_multimodal_content(content)
                result.append(
                    cast(ResponseInputItemParam, {"role": "developer", "content": content})
                )

            elif role == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    content = self._convert_multimodal_content(content)
                result.append(cast(ResponseInputItemParam, {"role": "user", "content": content}))

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
                    result.append(
                        cast(ResponseInputItemParam, {"role": "assistant", "content": content})
                    )

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

    def _from_specific_message(self, message: LLMSpecificMessage) -> ResponseInputItemParam:
        """Convert an OpenAI-Responses-specific message to an input item.

        Reasoning messages — persisted so the model's prior reasoning round-trips
        on later turns — become Responses ``reasoning`` input items. Anything
        else is assumed to already be in Responses input shape.

        Args:
            message: The LLM-specific message from the context.

        Returns:
            A Responses API input item.
        """
        payload = message.message
        if isinstance(payload, dict) and payload.get("type") == "reasoning":
            item: dict[str, Any] = {
                "type": "reasoning",
                "id": payload.get("id"),
                "summary": payload.get("summary", []),
            }
            encrypted = payload.get("encrypted_content")
            if encrypted:
                item["encrypted_content"] = encrypted
            return cast(ResponseInputItemParam, item)
        return cast(ResponseInputItemParam, payload)

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
