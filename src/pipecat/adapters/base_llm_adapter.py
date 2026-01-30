#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base adapter for LLM provider integration.

This module provides the abstract base class for implementing LLM provider-specific
adapters that handle tool format conversion and standardization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

from loguru import logger

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

    Subclasses must implement provider-specific conversion logic.
    """

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

        Args:
            tools: Tools in standard format or provider-specific format.

        Returns:
            List of tools converted to provider format, or original tools
            if not in standard format.
        """
        if isinstance(tools, ToolsSchema):
            logger.debug(f"Retrieving the tools using the adapter: {type(self)}")
            return self.to_provider_tools_format(tools)
        # Fallback to return the same tools in case they are not in a standard format
        return tools

    def _warn_about_orphaned_tool_messages(self, context: LLMContext) -> None:
        """Warn if context contains messages referencing tools that aren't currently available.

        This can happen when tools are removed/deactivated but the conversation history
        still contains function calls or tool responses for those tools. Such orphaned
        messages may cause API errors from the LLM provider.

        Args:
            context: The LLM context to check.
        """
        # Get the set of currently available tool names
        available_tool_names: set[str] = set()
        if isinstance(context.tools, ToolsSchema):
            available_tool_names = {tool.name for tool in context.tools.standard_tools}
            # Note: We don't check custom tools as they may have varying formats

        # Track orphaned function names found in messages
        orphaned_tools: set[str] = set()

        for message in self.get_messages(context):
            if isinstance(message, LLMSpecificMessage):
                # Skip LLM-specific messages for now
                continue

            # Check for tool_calls in assistant messages
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    func_name = tc.get("function", {}).get("name")
                    if func_name and available_tool_names and func_name not in available_tool_names:
                        orphaned_tools.add(func_name)

        # Log warning for orphaned messages
        if orphaned_tools:
            logger.warning(
                f"Context contains references to tools that are no longer available: "
                f"{sorted(orphaned_tools)}. This may cause unexpected behavior or API errors."
            )

    # TODO: we can move the logic to also handle the Messages here
