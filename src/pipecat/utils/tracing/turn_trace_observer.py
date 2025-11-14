#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


"""Turn trace observer for OpenTelemetry tracing in Pipecat.

This module provides an observer that creates trace spans for each conversation
turn, integrating with the turn tracking system to provide hierarchical tracing
of conversation flows.
"""

import time
from typing import TYPE_CHECKING, Dict, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    MetricsFrame,
    STTMuteFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.context import get_current_run_id
from pipecat.utils.tracing.context_registry import ContextProviderRegistry
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
        """Initialize the turn trace observer.

        Args:
            turn_tracker: The turn tracking observer to monitor.
            conversation_id: Optional conversation ID for grouping turns.
            additional_span_attributes: Additional attributes to add to spans.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._turn_tracker = turn_tracker
        self._current_span: Optional["Span"] = None
        self._current_turn_number: int = 0
        self._trace_context_map: Dict[int, "SpanContext"] = {}
        self._tracer = trace.get_tracer("pipecat.turn") if is_tracing_available() else None

        self._processed_frames = set()

        # Conversation tracking properties
        self._conversation_span: Optional["Span"] = None
        self._conversation_id = conversation_id
        self._additional_span_attributes = additional_span_attributes or {}

        # Get workflow run ID and providers from registry
        self._workflow_run_id = conversation_id or get_current_run_id()
        if self._workflow_run_id:
            # Ensure workflow_run_id is a string for consistency
            workflow_run_id_str = str(self._workflow_run_id)
            self._conversation_provider, self._turn_provider = (
                ContextProviderRegistry.get_or_create_providers(workflow_run_id_str)
            )
        else:
            # Fallback to singleton instances for backward compatibility
            self._conversation_provider = ConversationContextProvider.get_instance()
            self._turn_provider = TurnContextProvider.get_instance()

        # Latency measurement helpers (reset on every turn)
        self._latency_span: Optional["Span"] = None
        # Timestamp when the user was detected to stop speaking (after end of turn)
        self._user_stopped_ts: float = 0.0
        # Timestamp when VAD non-definitively detected the user stopped speaking (pre end of turn)
        self._vad_stopped_ts: float = 0.0

        # STT mute tracking
        self._stt_muted: bool = False

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

        Args:
            data: The frame push event data.
        """
        # If tracing is not available or no active turn span, do nothing
        if not (is_tracing_available() and self._current_span and self._tracer):
            return

        # Only process downstream frames
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        if data.frame.id in self._processed_frames:
            return
        self._processed_frames.add(data.frame.id)

        frame = data.frame

        # ------------------------------------------------------------
        # 1) Latency attributes within the pre-allocated span
        # ------------------------------------------------------------
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # Record the timestamp – actual span already exists from turn start
            # logger.debug("VADUserStoppedSpeakingFrame in TurnTraceObserver")
            self._vad_stopped_ts = time.time()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Record generic user stop speaking timestamp (may occur before definitive VAD stop)
            # logger.debug("UserStoppedSpeakingFrame in TurnTraceObserver")
            self._user_stopped_ts = time.time()

        elif isinstance(frame, BotStartedSpeakingFrame):
            # Capture latency attribute once
            # logger.debug("BotStartedSpeakingFrame in TurnTraceObserver")
            if self._latency_span is not None:
                now = time.time()

                # Latency from VAD definitive stop to bot start
                if self._vad_stopped_ts > 0:
                    latency_vad = max(now - self._vad_stopped_ts, 0.0)
                    self._latency_span.set_attribute(
                        "vad_stop_to_bot_start_latency", latency_vad * 1000
                    )

                # Latency from first user stop event to bot start
                if self._user_stopped_ts > 0:
                    latency_user = max(now - self._user_stopped_ts, 0.0)
                    self._latency_span.set_attribute(
                        "user_stop_to_bot_start_latency", latency_user * 1000
                    )

        # ------------------------------------------------------------
        # 2) MetricsFrames – capture TTFB and end-of-turn processing times
        # ------------------------------------------------------------
        if isinstance(frame, MetricsFrame) and self._latency_span is not None:
            for metric in frame.data:
                try:
                    processor_name = metric.processor.lower()
                except AttributeError:
                    processor_name = "unknown"

                # Time-to-first-byte metrics
                metric_type = metric.__class__.__name__

                if metric_type == "TTFBMetricsData":
                    # Store as milliseconds for consistency
                    self._latency_span.set_attribute(
                        f"{processor_name}.ttfb_ms", metric.value * 1000
                    )
                elif metric_type == "SmartTurnMetricsData":
                    # Detailed SmartTurn metrics
                    for attr_name in [
                        "e2e_processing_time_ms",
                        "inference_time_ms",
                        "server_total_time_ms",
                    ]:
                        if hasattr(metric, attr_name):
                            self._latency_span.set_attribute(
                                f"end_of_turn.{attr_name}", getattr(metric, attr_name)
                            )

    def start_conversation_tracing(self, conversation_id: Optional[str] = None):
        """Start a new conversation span.

        Args:
            conversation_id: Optional custom ID for the conversation. If None, a UUID will be generated.
        """
        if not is_tracing_available() or not self._tracer:
            return

        # Generate a conversation ID if not provided
        if conversation_id is None:
            conversation_id = self._conversation_provider.generate_conversation_id()
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
        self._conversation_provider.set_current_conversation_context(
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
            self._turn_provider.set_current_turn_context(None)

        # Now end the conversation span if it exists
        if self._conversation_span:
            # End the span
            self._conversation_span.end()
            self._conversation_span = None

            # Clear the context provider
            self._conversation_provider.set_current_conversation_context(None)

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
            parent_context = self._conversation_provider.get_current_conversation_context()

        # Create a new span for this turn
        self._current_span = self._tracer.start_span(f"turn-{turn_number}", context=parent_context)
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
        self._turn_provider.set_current_turn_context(self._current_span.get_span_context())

        # Pre-create latency span for this turn so we can accrue attributes over time
        self._latency_span = self._tracer.start_span(
            "latency.user_stop_to_bot_start", context=self._turn_provider.get_current_turn_context()
        )
        self._user_stopped_ts = 0.0
        self._vad_stopped_ts = 0.0

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

            # End latency span if it hasn't been closed yet (e.g., bot never spoke)
            if self._latency_span is not None:
                self._latency_span.end()
                self._latency_span = None
                self._user_stopped_ts = 0.0
                self._vad_stopped_ts = 0.0

            # Now end the main turn span
            self._current_span.end()
            self._current_span = None

            # Clear the context provider
            self._turn_provider.set_current_turn_context(None)

            logger.debug(f"Ended tracing for Turn {turn_number}")

    def get_current_turn_context(self) -> Optional["SpanContext"]:
        """Get the span context for the current turn.

        This can be used by services to create child spans.

        Returns:
            The current turn's span context or None if not available.
        """
        if not is_tracing_available() or not self._current_span:
            return None

        return self._current_span.get_span_context()

    def get_turn_context(self, turn_number: int) -> Optional["SpanContext"]:
        """Get the span context for a specific turn.

        This can be used by services to create child spans.

        Args:
            turn_number: The turn number to get context for.

        Returns:
            The specified turn's span context or None if not available.
        """
        if not is_tracing_available():
            return None

        return self._trace_context_map.get(turn_number)
