#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline-scoped tracing context for OpenTelemetry tracing in Pipecat.

This module provides a per-pipeline tracing context that holds the current
conversation and turn span contexts. Each PipelineTask creates its own
TracingContext, ensuring concurrent pipelines do not interfere with each other.
"""

import uuid
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.trace import SpanContext

from pipecat.utils.tracing.setup import is_tracing_available

if is_tracing_available():
    from opentelemetry.context import Context
    from opentelemetry.trace import NonRecordingSpan, SpanContext, set_span_in_context


class TracingContext:
    """Pipeline-scoped tracing context.

    Holds the current conversation and turn span contexts for a single pipeline.
    Created by PipelineTask, passed to TurnTraceObserver (writer) and services
    (readers) via StartFrame.
    """

    def __init__(self):
        """Initialize the tracing context with empty state."""
        self._conversation_context: Optional["Context"] = None
        self._turn_context: Optional["Context"] = None
        self._conversation_id: Optional[str] = None

    def set_conversation_context(
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
            non_recording_span = NonRecordingSpan(span_context)
            self._conversation_context = set_span_in_context(non_recording_span)
        else:
            self._conversation_context = None

    def get_conversation_context(self) -> Optional["Context"]:
        """Get the OpenTelemetry context for the current conversation.

        Returns:
            The current conversation context or None if not available.
        """
        return self._conversation_context

    def set_turn_context(self, span_context: Optional["SpanContext"]):
        """Set the current turn context.

        Args:
            span_context: The span context for the current turn or None to clear it.
        """
        if not is_tracing_available():
            return

        if span_context:
            non_recording_span = NonRecordingSpan(span_context)
            self._turn_context = set_span_in_context(non_recording_span)
        else:
            self._turn_context = None

    def get_turn_context(self) -> Optional["Context"]:
        """Get the OpenTelemetry context for the current turn.

        Returns:
            The current turn context or None if not available.
        """
        return self._turn_context

    @property
    def conversation_id(self) -> Optional[str]:
        """Get the ID for the current conversation.

        Returns:
            The current conversation ID or None if not available.
        """
        return self._conversation_id

    @staticmethod
    def generate_conversation_id() -> str:
        """Generate a new conversation ID.

        Returns:
            A new randomly generated UUID string.
        """
        return str(uuid.uuid4())
