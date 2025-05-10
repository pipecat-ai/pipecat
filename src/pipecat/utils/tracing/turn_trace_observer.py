#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Dict, Optional

from loguru import logger

from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.utils.tracing.context_provider import TurnContextProvider
from pipecat.utils.tracing.tracing import is_tracing_available

if is_tracing_available():
    from opentelemetry import trace
    from opentelemetry.trace import Span, SpanContext


class TurnTraceObserver(BaseObserver):
    """Observer that creates trace spans for each conversation turn.

    This observer uses TurnTrackingObserver to track turns and creates
    OpenTelemetry spans for each turn. Service spans (STT, LLM, TTS)
    become children of the turn spans.
    """

    def __init__(self, turn_tracker: TurnTrackingObserver):
        super().__init__()
        self._turn_tracker = turn_tracker
        self._current_span: Optional[Span] = None
        self._current_turn_number: int = 0
        self._trace_context_map: Dict[int, SpanContext] = {}
        self._tracer = trace.get_tracer("pipecat.turn") if is_tracing_available() else None

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

    async def _handle_turn_started(self, turn_number: int):
        """Handle a turn start event by creating a new span."""
        if not is_tracing_available() or not self._tracer:
            return

        # Create a new span for this turn
        self._current_span = self._tracer.start_span(f"Turn {turn_number}")
        self._current_turn_number = turn_number

        # Set span attributes
        self._current_span.set_attribute("turn.number", turn_number)
        self._current_span.set_attribute("turn.type", "conversation")

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

    def get_current_turn_context(self) -> Optional[SpanContext]:
        """Get the span context for the current turn.

        This can be used by services to create child spans.
        """
        if not is_tracing_available() or not self._current_span:
            return None

        return self._current_span.get_span_context()

    def get_turn_context(self, turn_number: int) -> Optional[SpanContext]:
        """Get the span context for a specific turn.

        This can be used by services to create child spans.
        """
        if not is_tracing_available():
            return None

        return self._trace_context_map.get(turn_number)
