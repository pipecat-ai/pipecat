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
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

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
    - Resolving conflicts between ``system_instruction`` and initial
      system/developer messages in the conversation context.

    Subclasses must implement provider-specific conversion logic.
    """

    def __init__(self):
        """Initialize the adapter."""
        self._warned_system_instruction = False

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

    def _extract_initial_system_or_developer(
        self,
        messages: list,
        *,
        system_instruction: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract an initial system/developer message for use as a system instruction.

        Only useful for services that expect the system instruction as a
        separate parameter, not inline in conversation history (today, all
        non-OpenAI services).

        Checks ``messages[0]``. Behavior:

        - ``"system"`` role: assumed to be intended as the system instruction.
          Extract (pop from messages).
        - ``"developer"`` role **without** ``system_instruction``: also assumed
          to be intended as the system instruction. Extract (pop).
        - ``"developer"`` role **with** ``system_instruction``: assumed to be
          intended as a conversation-history message (since a system instruction
          is already provided). Don't extract; convert to ``"user"`` in-place.
        - Any other role: no-op.

        If extracting would leave the messages list empty
        (``len(messages) == 1``), the message is converted to ``"user"`` role
        instead of being extracted. This prevents sending an empty conversation
        history to providers that require at least one non-system message.

        Args:
            messages: Message list in standard format (mutated in-place).
            system_instruction: The system instruction from service settings
                or ``run_inference``, used to decide whether to extract a
                ``"developer"`` message.

        Returns:
            ``(extracted_content, original_role)`` where *original_role* is
            ``"system"`` or ``"developer"``, or ``(None, None)`` if nothing
            was extracted.
        """
        if not messages:
            return None, None

        role = messages[0].get("role")
        if role not in ("system", "developer"):
            return None, None

        # "developer" + system_instruction present → keep in messages as "user"
        if role == "developer" and system_instruction:
            messages[0]["role"] = "user"
            return None, None

        # Would extracting empty the list? Convert to "user" instead.
        if len(messages) == 1:
            messages[0]["role"] = "user"
            return None, None

        # Extract
        content = messages[0].get("content", "")
        if isinstance(content, list):
            # Join text parts for providers that expect a string system instruction
            content = " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        messages.pop(0)
        return content, role

    def _resolve_system_instruction(
        self,
        initial_context_message: Optional[str],
        initial_context_message_role: Optional[str],
        system_instruction: Optional[str],
        *,
        discard_context_system: bool,
    ) -> Optional[str]:
        """Resolve conflict between ``system_instruction`` and an initial context message.

        Only warns when *initial_context_message_role* is ``"system"`` (not
        ``"developer"``), since a developer message coexisting with
        ``system_instruction`` is expected and handled elsewhere.

        Args:
            initial_context_message: Content extracted from ``messages[0]``
                by :meth:`_extract_initial_system_or_developer`, or detected
                inline (OpenAI adapters).
            initial_context_message_role: ``"system"`` or ``"developer"`` —
                the original role before extraction/detection.
            system_instruction: From service settings or ``run_inference`` param.
            discard_context_system: If ``True`` (non-OpenAI adapters), the
                context system message is discarded when ``system_instruction``
                is also present. If ``False`` (OpenAI adapters), both are kept.

        Returns:
            The effective system instruction to use, or ``None`` if the system
            instruction is already represented in the messages (OpenAI path).
        """
        both_present = initial_context_message and system_instruction
        from_system_role = initial_context_message_role == "system"

        if both_present and from_system_role:
            if not self._warned_system_instruction:
                self._warned_system_instruction = True
                if discard_context_system:
                    logger.warning(
                        "Both system_instruction and a system message in context are set."
                        " Using system_instruction."
                    )
                else:
                    logger.warning(
                        "Both system_instruction and an initial system message in context"
                        " are set, which may be unintended. Prefer system_instruction."
                    )

        if system_instruction:
            if discard_context_system:
                return system_instruction
            else:
                # OpenAI path: caller prepends; return the instruction for prepending
                return system_instruction

        if initial_context_message:
            if discard_context_system:
                return initial_context_message
            else:
                # Content is already in messages; nothing to prepend
                return None

        return None
