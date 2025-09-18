#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Registry for managing context providers per workflow run.

This module provides a registry that maintains separate conversation and turn
context providers for each workflow run, enabling proper trace isolation
between concurrent conversations.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from pipecat.utils.tracing.conversation_context_provider import ConversationContextProvider
from pipecat.utils.tracing.turn_context_provider import TurnContextProvider

# Import types for type checking only
if TYPE_CHECKING:
    from opentelemetry.context import Context

from pipecat.utils.context import get_current_run_id


class ContextProviderRegistry:
    """Registry for managing context providers per workflow run.

    This registry maintains separate conversation and turn context providers
    for each workflow run, enabling proper trace isolation between concurrent
    conversations.
    """

    _instance = None

    def __init__(self):
        """Initialize the registry with an empty providers dictionary."""
        self._providers: Dict[str, Tuple["ConversationContextProvider", "TurnContextProvider"]] = {}

    @classmethod
    def get_instance(cls):
        """Get the singleton registry instance.

        Returns:
            The singleton ContextProviderRegistry instance.
        """
        if cls._instance is None:
            cls._instance = ContextProviderRegistry()
        return cls._instance

    @classmethod
    def get_or_create_providers(
        cls, workflow_run_id: str
    ) -> Tuple["ConversationContextProvider", "TurnContextProvider"]:
        """Get or create context providers for a specific workflow run.

        Args:
            workflow_run_id: The ID of the workflow run.

        Returns:
            A tuple of (ConversationContextProvider, TurnContextProvider) for the workflow.
        """
        registry = cls.get_instance()
        if workflow_run_id not in registry._providers:
            registry._providers[workflow_run_id] = (
                ConversationContextProvider(),
                TurnContextProvider(),
            )
        return registry._providers[workflow_run_id]

    @classmethod
    def remove_providers(cls, workflow_run_id: str):
        """Remove context providers for a specific workflow run.

        This should be called when a workflow run completes to prevent memory leaks.

        Args:
            workflow_run_id: The ID of the workflow run to clean up.
        """
        registry = cls.get_instance()
        registry._providers.pop(workflow_run_id, None)

    @classmethod
    def clear_all(cls):
        """Clear all registered providers. Mainly used for testing."""
        registry = cls.get_instance()
        registry._providers.clear()


def get_current_conversation_context(workflow_run_id: Optional[str] = None) -> Optional["Context"]:
    """Get the OpenTelemetry context for the current conversation.

    Args:
        workflow_run_id: Optional workflow run ID. If not provided, uses the current context.

    Returns:
        The current conversation context or None if not available.
    """
    # Use provided workflow_run_id or get from context
    if workflow_run_id is None:
        workflow_run_id = get_current_run_id()

    if workflow_run_id is None:
        # Backward compatibility: return default singleton-like behavior
        provider = ConversationContextProvider.get_instance()
        return provider.get_current_conversation_context()

    conversation_provider, _ = ContextProviderRegistry.get_or_create_providers(workflow_run_id)

    return conversation_provider.get_current_conversation_context()


def get_current_turn_context(workflow_run_id: Optional[str] = None) -> Optional["Context"]:
    """Get the OpenTelemetry context for the current turn.

    Args:
        workflow_run_id: Optional workflow run ID. If not provided, uses the current context.

    Returns:
        The current turn context or None if not available.
    """
    # Use provided workflow_run_id or get from context
    if workflow_run_id is None:
        workflow_run_id = get_current_run_id()

    if workflow_run_id is None:
        # Backward compatibility: return default singleton-like behavior
        provider = TurnContextProvider.get_instance()
        return provider.get_current_turn_context()

    _, turn_provider = ContextProviderRegistry.get_or_create_providers(workflow_run_id)

    return turn_provider.get_current_turn_context()
