#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base adapter for LLM provider integration.

This module provides the abstract base class for implementing LLM provider-specific
adapters that handle tool format conversion and standardization.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    NotGiven,
)

# Should be a TypedDict
TLLMInvocationParams = TypeVar("TLLMInvocationParams", bound=dict[str, Any])


class BaseLLMAdapter(ABC, Generic[TLLMInvocationParams]):
    """Abstract base class for LLM provider adapters.

    Provides a standard interface for converting to provider-specific formats.

    Handles:

    - Extracting provider-specific parameters for LLM invocation from a
      universal LLM context
    - Converting standardized tools schema to provider-specific tool formats.
    - Extracting messages from the LLM context for the purposes of logging
      about the specific provider.
    - Resolving conflicts between ``system_instruction`` and initial
      system/developer messages in the conversation context.

    Subclasses must implement provider-specific conversion logic.
    """

    def __init__(self):
        """Initialize the adapter."""
        self._warned_system_instruction = False
        self._builtin_tools: Dict[str, FunctionSchema] = {}

    @property
    def builtin_tools(self) -> Dict[str, FunctionSchema]:
        """Built-in tools automatically merged into every inference request.

        Keyed by tool name for O(1) lookup, insertion, and removal.  The
        service injects tools here so they are sent transparently on every
        inference request without the user having to add them to their
        ``ToolsSchema``.

        Returns:
            Mutable dict mapping tool name to ``FunctionSchema``.
        """
        return self._builtin_tools

    @property
    @abstractmethod
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for this LLM provider.

        Returns:
            The identifier string.
        """
        pass

    @abstractmethod
    def get_llm_invocation_params(self, context: LLMContext, **kwargs) -> TLLMInvocationParams:
        """Get provider-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.
            **kwargs: Additional provider-specific arguments that subclasses can use.

        Returns:
            Provider-specific parameters for invoking the LLM.
        """
        pass

    @abstractmethod
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Any]:
        """Convert tools schema to the provider's specific format.

        Args:
            tools_schema: The standardized tools schema to convert.

        Returns:
            List of tools in the provider's expected format.
        """
        pass

    @abstractmethod
    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about this provider.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about this
            provider.
        """
        pass

    def create_llm_specific_message(self, message: Any) -> LLMSpecificMessage:
        """Create an LLM-specific message (as opposed to a standard message) for use in an LLMContext.

        Args:
            message: The message content.

        Returns:
            A LLMSpecificMessage instance.
        """
        return LLMSpecificMessage(llm=self.id_for_llm_specific_messages, message=message)

    def get_messages(self, context: LLMContext) -> List[LLMContextMessage]:
        """Get messages from the LLM context, including standard and LLM-specific messages.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages including standard and LLM-specific messages.
        """
        return context.get_messages(self.id_for_llm_specific_messages)

    def from_standard_tools(self, tools: Any) -> List[Any] | NotGiven:
        """Convert tools from standard format to provider format.

        Built-in tools are automatically merged into the schema before conversion so that every
        inference request receives them without the user having to declare them explicitly.

        Args:
            tools: Tools in standard format or provider-specific format.

        Returns:
            List of tools converted to provider format, or original tools
            if not in standard format.
        """
        if self._builtin_tools:
            if isinstance(tools, ToolsSchema):
                tools = ToolsSchema(
                    standard_tools=tools.standard_tools + list(self._builtin_tools.values()),
                    custom_tools=tools.custom_tools,
                )
            else:
                # User supplied tools in a legacy/provider-specific format.
                # Built-in tools cannot be safely merged, so they will not be injected.
                # Migrate to ToolsSchema to enable built-in tool support; use custom_tools
                # as an escape hatch for any provider-specific tools that don't fit the
                # standard schema.
                if tools is not None:
                    warnings.warn(
                        "Built-in tools (e.g. async tool cancellation) could not be injected "
                        "because the supplied tools are not a ToolsSchema instance. "
                        "Migrate to ToolsSchema to enable built-in tool support. "
                        "Use ToolsSchema(custom_tools=...) as an escape hatch for any "
                        "provider-specific tools that don't fit the standard schema.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                # Fall through and return the original tools unchanged.

        if isinstance(tools, ToolsSchema):
            return self.to_provider_tools_format(tools)
        # Fallback to return the same tools in case they are not in a standard format
        return tools

    def _extract_initial_system(
        self,
        messages: list,
        *,
        system_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Extract an initial ``"system"`` message for use as a system instruction.

        Only useful for services that expect the system instruction as a
        separate parameter, not inline in conversation history (today, all
        non-OpenAI services). Does not extract ``"developer"`` messages —
        those are converted to ``"user"`` by the adapter's subsequent message
        loop, like any other non-system role the provider doesn't support.

        Checks ``messages[0]``. If the role is ``"system"``, pops and returns
        its content. If extracting would leave the messages list empty
        (``len(messages) == 1``), the message is converted to ``"user"``
        role instead of being extracted, to prevent sending an empty
        conversation history to providers that require at least one
        non-system message.

        Args:
            messages: Message list in standard format (mutated in-place).
            system_instruction: The system instruction from service settings
                or ``run_inference``. Only used to decide whether to warn
                about a conflict in the single-message case.

        Returns:
            The extracted system message content, or ``None`` if nothing
            was extracted.
        """
        if not messages:
            return None

        if messages[0].get("role") != "system":
            return None

        # Would extracting empty the list? Convert to "user" instead.
        if len(messages) == 1:
            if system_instruction:
                if not self._warned_system_instruction:
                    self._warned_system_instruction = True
                    logger.warning(
                        "Both system_instruction and an initial system message in"
                        " context are set. Using system_instruction. The context"
                        " system message is being converted to a user message to"
                        " avoid sending an empty conversation history."
                    )
            messages[0]["role"] = "user"
            return None

        # Extract
        content = messages[0].get("content", "")
        if isinstance(content, list):
            # Join text parts for providers that expect a string system instruction
            content = " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        messages.pop(0)
        return content

    def _resolve_system_instruction(
        self,
        system_from_context: Optional[str],
        system_instruction: Optional[str],
        *,
        discard_context_system: bool,
    ) -> Optional[str]:
        """Resolve conflict between ``system_instruction`` and an extracted context system message.

        Args:
            system_from_context: Content extracted from an initial ``"system"``
                message by :meth:`_extract_initial_system`, or detected
                inline (OpenAI adapters).
            system_instruction: From service settings or ``run_inference`` param.
            discard_context_system: If ``True`` (non-OpenAI adapters), the
                context system message is discarded when ``system_instruction``
                is also present. If ``False`` (OpenAI adapters), both are kept.

        Returns:
            The effective system instruction to use, or ``None`` if the system
            instruction is already represented in the messages (OpenAI path).
        """
        if system_from_context and system_instruction:
            if not self._warned_system_instruction:
                self._warned_system_instruction = True
                if discard_context_system:
                    logger.warning(
                        "Both system_instruction and an initial system message"
                        " in context are set. Using system_instruction."
                    )
                else:
                    logger.warning(
                        "Both system_instruction and an initial system message"
                        " in context are set, which may be unintended. Keeping"
                        " both, but consider using system_instruction for"
                        " system-level instructions and developer messages in"
                        " context for supplementary guidance."
                    )

        if system_instruction:
            return system_instruction

        if system_from_context:
            if discard_context_system:
                return system_from_context
            else:
                # Content is already in messages; nothing to prepend
                return None

        return None
