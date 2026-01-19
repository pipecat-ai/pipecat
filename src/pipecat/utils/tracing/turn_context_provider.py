#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn context provider for OpenTelemetry tracing in Pipecat.

This module provides a turn context provider that manages turn tracing contexts,
allowing services to create child spans that are properly associated with their
specific conversation turn.
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


class TurnContextProvider:
    """Provides access to a turn's tracing context.

    This class manages the tracing context for a specific turn.
    Multiple instances can exist to support concurrent conversations.
    """

    # Keep singleton pattern for backward compatibility
    _instance = None

    def __init__(self):
        """Initialize a new turn context provider."""
        self._current_turn_context: Optional["Context"] = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance (for backward compatibility).

        Returns:
            The singleton TurnContextProvider instance.
        """
        if cls._instance is None:
            cls._instance = TurnContextProvider()
        return cls._instance

    def set_current_turn_context(self, span_context: Optional["SpanContext"]):
        """Set the current turn context.

        Args:
            span_context: The span context for the current turn or None to clear it.
        """
        if not is_tracing_available():
            return

        if span_context:
            # Create a non-recording span from the span context
            non_recording_span = NonRecordingSpan(span_context)
            self._current_turn_context = set_span_in_context(non_recording_span)
        else:
            self._current_turn_context = None

    def get_current_turn_context(self) -> Optional["Context"]:
        """Get the OpenTelemetry context for the current turn.

        Returns:
            The current turn context or None if not available.
        """
        return self._current_turn_context
