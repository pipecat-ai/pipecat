#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import TYPE_CHECKING, Dict, Optional

from loguru import logger

from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.utils.tracing.conversation_context_provider import ConversationContextProvider
from pipecat.utils.tracing.setup import is_tracing_available
from pipecat.utils.tracing.turn_context_provider import TurnContextProvider

# Import types for type checking only
if TYPE_CHECKING:
    from opentelemetry.trace import Span, SpanContext

if is_tracing_available():
    from opentelemetry import trace
    from opentelemetry.trace import Span, SpanContext


class TurnTraceObserver(BaseObserver):
    """Observer that creates trace spans for each conversation turn.

    This observer uses TurnTrackingObserver to track turns and creates
    OpenTelemetry spans for each turn. Service spans (STT, LLM, TTS)
    become children of the turn spans.

    If conversation tracing is enabled, turns become children of a
    conversation span that encapsulates the entire session.
    """

    def __init__(
        self,
        turn_tracker: TurnTrackingObserver,
        conversation_id: Optional[str] = None,
        additional_span_attributes: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._turn_tracker = turn_tracker
        self._current_span: Optional["Span"] = None
        self._current_turn_number: int = 0
        self._trace_context_map: Dict[int, "SpanContext"] = {}
        self._tracer = trace.get_tracer("pipecat.turn") if is_tracing_available() else None

        # Conversation tracking properties
        self._conversation_span: Optional["Span"] = None
        self._conversation_id = conversation_id
        self._additional_span_attributes = additional_span_attributes or {}

        if turn_tracker:

            @turn_tracker.event_handler("on_turn_started")
            async def on_turn_started(tracker, turn_number):
                await self._handle_turn_started(turn_number)

            @turn_tracker.event_handler("on_turn_ended")
            async def on_turn_ended(tracker, turn_number, duration, was_interrupted):
                await self._handle_turn_ended(turn_number, duration, was_interrupted)

    async def on_push_frame(self, data: FramePushed):
        """Process a frame without modifying it.

        This observer doesn't need to process individual frames as it
        relies on turn start/end events from the turn tracker.
        """
        pass

    def start_conversation_tracing(self, conversation_id: Optional[str] = None):
        """Start a new conversation span.

        Args:
            conversation_id: Optional custom ID for the conversation. If None, a UUID will be generated.
        """
        if not is_tracing_available() or not self._tracer:
            return

        # Generate a conversation ID if not provided
        context_provider = ConversationContextProvider.get_instance()
        if conversation_id is None:
            conversation_id = context_provider.generate_conversation_id()
            logger.debug(f"Generated new conversation ID: {conversation_id}")

        self._conversation_id = conversation_id

        # Create a new span for this conversation
        self._conversation_span = self._tracer.start_span("conversation")

        # Set span attributes
        self._conversation_span.set_attribute("conversation.id", conversation_id)
        self._conversation_span.set_attribute("conversation.type", "voice")
        # Set custom otel attributes if provided
        for k, v in (self._additional_span_attributes or {}).items():
            self._conversation_span.set_attribute(k, v)

        # Update the conversation context provider
        context_provider.set_current_conversation_context(
            self._conversation_span.get_span_context(), conversation_id
        )

        logger.debug(f"Started tracing for Conversation {conversation_id}")

    def end_conversation_tracing(self):
        """End the current conversation span and ensure the last turn is closed."""
        if not is_tracing_available():
            return

        # First, ensure any active turn is closed properly
        if self._current_span:
            # If we have an active turn span, end it with a standard duration
            logger.debug(f"Ending Turn {self._current_turn_number} due to conversation end")
            self._current_span.set_attribute("turn.was_interrupted", True)
            self._current_span.set_attribute("turn.ended_by_conversation_end", True)
            self._current_span.end()
            self._current_span = None

            # Clear the turn context provider
            context_provider = TurnContextProvider.get_instance()
            context_provider.set_current_turn_context(None)

        # Now end the conversation span if it exists
        if self._conversation_span:
            # End the span
            self._conversation_span.end()
            self._conversation_span = None

            # Clear the context provider
            context_provider = ConversationContextProvider.get_instance()
            context_provider.set_current_conversation_context(None)

            logger.debug(f"Ended tracing for Conversation {self._conversation_id}")
            self._conversation_id = None

    async def _handle_turn_started(self, turn_number: int):
        """Handle a turn start event by creating a new span."""
        if not is_tracing_available() or not self._tracer:
            return

        # If this is the first turn and no conversation span exists yet,
        # start the conversation tracing (will generate ID if needed)
        if turn_number == 1 and not self._conversation_span:
            self.start_conversation_tracing(self._conversation_id)

        # Get the parent context - conversation if available, otherwise use root context
        parent_context = None
        if self._conversation_span:
            context_provider = ConversationContextProvider.get_instance()
            parent_context = context_provider.get_current_conversation_context()

        # Create a new span for this turn
        self._current_span = self._tracer.start_span("turn", context=parent_context)
        self._current_turn_number = turn_number

        # Set span attributes
        self._current_span.set_attribute("turn.number", turn_number)
        self._current_span.set_attribute("turn.type", "conversation")

        # Add conversation ID attribute if available
        if self._conversation_id:
            self._current_span.set_attribute("conversation.id", self._conversation_id)

        # Store the span context so services can become children of this span
        self._trace_context_map[turn_number] = self._current_span.get_span_context()

        # Update the context provider so services can access this span
        context_provider = TurnContextProvider.get_instance()
        context_provider.set_current_turn_context(self._current_span.get_span_context())

        logger.debug(f"Started tracing for Turn {turn_number}")

    async def _handle_turn_ended(self, turn_number: int, duration: float, was_interrupted: bool):
        """Handle a turn end event by ending the current span."""
        if not is_tracing_available() or not self._current_span:
            return

        # Only end the span if it matches the current turn
        if turn_number == self._current_turn_number:
            # Set additional attributes
            self._current_span.set_attribute("turn.duration_seconds", duration)
            self._current_span.set_attribute("turn.was_interrupted", was_interrupted)

            # End the span
            self._current_span.end()
            self._current_span = None

            # Clear the context provider
            context_provider = TurnContextProvider.get_instance()
            context_provider.set_current_turn_context(None)

            logger.debug(f"Ended tracing for Turn {turn_number}")

    def get_current_turn_context(self) -> Optional["SpanContext"]:
        """Get the span context for the current turn.

        This can be used by services to create child spans.
        """
        if not is_tracing_available() or not self._current_span:
            return None

        return self._current_span.get_span_context()

    def get_turn_context(self, turn_number: int) -> Optional["SpanContext"]:
        """Get the span context for a specific turn.

        This can be used by services to create child spans.
        """
        if not is_tracing_available():
            return None

        return self._trace_context_map.get(turn_number)
