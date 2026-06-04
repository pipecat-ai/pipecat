#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM adapter for Pipecat."""

from typing import Any, TypedDict, TypeGuard, TypeVar, cast

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
    LLMStandardMessage,
    NotGiven,
)

_T = TypeVar("_T")


def _openai_from_llm_context_tool_choice(
    tool_choice: LLMContextToolChoice | NotGiven,
) -> ChatCompletionToolChoiceOptionParam | OpenAINotGiven:
    """Reinterpret an LLMContext ``tool_choice`` as OpenAI's type.

    The underlying types are currently aliased — ``LLMContextToolChoice`` is
    ``ChatCompletionToolChoiceOptionParam`` and LLMContext's ``NotGiven`` is
    OpenAI's — so this is a typed no-op today. It's kept as a named boundary
    so that if the LLMContext side ever diverges from OpenAI's types, every
    crossing is visible and easy to update.
    """
    return cast("ChatCompletionToolChoiceOptionParam | OpenAINotGiven", tool_choice)


def _openai_from_llm_standard_message(
    message: LLMStandardMessage,
) -> ChatCompletionMessageParam:
    """Reinterpret an LLMContext standard message as OpenAI's type.

    Same rationale as :func:`_openai_from_llm_context_tool_choice`: the
    aliased types make this a no-op today, but the boundary is preserved
    for future divergence.
    """
    return cast("ChatCompletionMessageParam", message)


def is_given(value: _T | OpenAINotGiven) -> TypeGuard[_T]:
    """Check whether a value was explicitly provided.

    Typically used when checking whether a parameter or field typed with
    OpenAI's ``NotGiven`` was set::

        if is_given(tool_choice):
            ...

    Also acts as a type guard: inside a true branch, the value is narrowed
    to exclude ``OpenAINotGiven`` (e.g.
    ``ChatCompletionToolChoiceOptionParam | OpenAINotGiven`` becomes
    ``ChatCompletionToolChoiceOptionParam``).

    Args:
        value: The value to check.

    Returns:
        ``True`` if *value* is anything other than ``NOT_GIVEN``.
    """
    return not isinstance(value, OpenAINotGiven)


class OpenAILLMInvocationParams(TypedDict):
    """Context-based parameters for invoking OpenAI ChatCompletion API."""

    messages: list[ChatCompletionMessageParam]
    tools: list[ChatCompletionToolParam] | OpenAINotGiven
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
        system_instruction: str | None = None,
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
            # Detect initial system message for warning purposes (don't extract).
            # ChatCompletionMessageParam.content is `str | Iterable[...]`; we
            # only forward it for warning purposes, so coerce non-strings to
            # None — the resolver handles None.
            initial_content: str | None = None
            if messages and messages[0].get("role") == "system":
                raw_content = messages[0].get("content", "")
                if isinstance(raw_content, str):
                    initial_content = raw_content
            self._resolve_system_instruction(
                initial_content,
                system_instruction,
                discard_context_system=False,
            )
            messages = [{"role": "system", "content": system_instruction}] + messages

        return cast(
            OpenAILLMInvocationParams,
            {
                "messages": messages,
                # NOTE; LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
                "tools": self.from_standard_tools(context.tools),
                "tool_choice": _openai_from_llm_context_tool_choice(context.tool_choice),
            },
        )

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[ChatCompletionToolParam]:
        """Convert function schemas to OpenAI's function-calling format.

        Args:
            tools_schema: The Pipecat tools schema to convert.

        Returns:
            List of OpenAI formatted function call definitions ready for use
            with ChatCompletion API.
        """
        functions_schema = tools_schema.standard_tools
        # `function=...` expects a `FunctionDefinition` TypedDict; the dict
        # produced by `to_default_dict()` is structurally compatible. Cast at
        # the boundary.
        formatted_standard_tools: list[ChatCompletionToolParam] = [
            ChatCompletionToolParam(type="function", function=cast(Any, func.to_default_dict()))
            for func in functions_schema
        ]
        custom_openai_tools: list[ChatCompletionToolParam] = []
        if tools_schema.custom_tools:
            custom_openai_tools = cast(
                list[ChatCompletionToolParam],
                tools_schema.custom_tools.get(AdapterType.OPENAI, []),
            )
        return formatted_standard_tools + custom_openai_tools

    def get_messages_for_logging(self, context: LLMContext) -> list[dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about OpenAI.

        Binary data (images, audio) is replaced with short placeholders.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about OpenAI.
        """
        return cast(
            list[dict[str, Any]],
            self.get_messages(context, truncate_large_values=True),
        )

    def _from_universal_context_messages(
        self,
        messages: list[LLMContextMessage],
        *,
        convert_developer_to_user: bool,
    ) -> list[ChatCompletionMessageParam]:
        result = []
        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                # Extract the actual message content from LLMSpecificMessage
                result.append(message.message)
            else:
                # Standard message, pass through unchanged
                result.append(_openai_from_llm_standard_message(message))

        if convert_developer_to_user:
            for msg in result:
                if msg.get("role") == "developer":
                    msg["role"] = "user"

        return result

    def _from_standard_tool_choice(
        self, tool_choice: LLMContextToolChoice | NotGiven
    ) -> ChatCompletionToolChoiceOptionParam | OpenAINotGiven:
        return _openai_from_llm_context_tool_choice(tool_choice)
