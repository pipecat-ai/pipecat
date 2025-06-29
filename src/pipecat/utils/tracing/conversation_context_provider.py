#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Conversation context provider for OpenTelemetry tracing in Pipecat.

This module provides a singleton context provider that manages the current
conversation's tracing context, allowing services to create child spans
that are properly associated with the conversation.
"""

import uuid
from typing import TYPE_CHECKING, Optional

# Import types for type checking only
if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.trace import SpanContext

from pipecat.utils.tracing.setup import is_tracing_available

if is_tracing_available():
    from opentelemetry.context import Context
    from opentelemetry.trace import NonRecordingSpan, SpanContext, set_span_in_context


class ConversationContextProvider:
    """Provides access to the current conversation's tracing context.

    This is a singleton that can be used to get the current conversation's
    span context to create child spans (like turns).
    """

    _instance = None
    _current_conversation_context: Optional["Context"] = None
    _conversation_id: Optional[str] = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance.

        Returns:
            The singleton ConversationContextProvider instance.
        """
        if cls._instance is None:
            cls._instance = ConversationContextProvider()
        return cls._instance

    def set_current_conversation_context(
        self, span_context: Optional["SpanContext"], conversation_id: Optional[str] = None
    ):
        """Set the current conversation context.

        Args:
            span_context: The span context for the current conversation or None to clear it.
            conversation_id: Optional ID for the conversation.
        """
        if not is_tracing_available():
            return

        self._conversation_id = conversation_id

        if span_context:
            # Create a non-recording span from the span context
            non_recording_span = NonRecordingSpan(span_context)
            self._current_conversation_context = set_span_in_context(non_recording_span)
        else:
            self._current_conversation_context = None

    def get_current_conversation_context(self) -> Optional["Context"]:
        """Get the OpenTelemetry context for the current conversation.

        Returns:
            The current conversation context or None if not available.
        """
        return self._current_conversation_context

    def get_conversation_id(self) -> Optional[str]:
        """Get the ID for the current conversation.

        Returns:
            The current conversation ID or None if not available.
        """
        return self._conversation_id

    def generate_conversation_id(self) -> str:
        """Generate a new conversation ID.

        Returns:
            A new randomly generated UUID string.
        """
        return str(uuid.uuid4())


def get_current_conversation_context() -> Optional["Context"]:
    """Get the OpenTelemetry context for the current conversation.

    Returns:
        The current conversation context or None if not available.
    """
    provider = ConversationContextProvider.get_instance()
    return provider.get_current_conversation_context()


def get_conversation_id() -> Optional[str]:
    """Get the ID for the current conversation.

    Returns:
        The current conversation ID or None if not available.
    """
    provider = ConversationContextProvider.get_instance()
    return provider.get_conversation_id()
