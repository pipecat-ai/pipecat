#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import threading
import unittest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.tracing.tracing_context import TracingContext
from pipecat.utils.tracing.turn_trace_observer import TurnTraceObserver

if HAS_OPENTELEMETRY:

    class _InMemorySpanExporter(SpanExporter):
        """Simple in-memory span exporter for testing."""

        def __init__(self):
            """Initialize the exporter."""
            self._spans = []
            self._lock = threading.Lock()

        def export(self, spans):
            """Export spans to memory."""
            with self._lock:
                self._spans.extend(spans)
            return SpanExportResult.SUCCESS

        def get_finished_spans(self):
            """Return collected spans."""
            with self._lock:
                return list(self._spans)

        def clear(self):
            """Clear collected spans."""
            with self._lock:
                self._spans.clear()


@unittest.skipUnless(HAS_OPENTELEMETRY, "opentelemetry not installed")
class TestTurnTraceObserver(unittest.IsolatedAsyncioTestCase):
    """Tests for TurnTraceObserver."""

    def setUp(self):
        """Set up a fresh provider and exporter for each test.

        We create a dedicated TracerProvider per test and inject its tracer
        directly into the observer, avoiding the global provider singleton.
        """
        self._exporter = _InMemorySpanExporter()
        self._provider = TracerProvider()
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._tracer = self._provider.get_tracer("pipecat.turn")

    def tearDown(self):
        """Shut down the provider to flush spans."""
        self._provider.shutdown()

    def _create_observers(self, conversation_id=None, tracing_context=None):
        """Create a standard set of turn/trace observers.

        Args:
            conversation_id: Optional conversation ID.
            tracing_context: Optional TracingContext instance.

        Returns:
            Tuple of (turn_tracker, latency_tracker, trace_observer, tracing_context).
        """
        tracing_context = tracing_context or TracingContext()
        turn_tracker = TurnTrackingObserver(turn_end_timeout_secs=0.2)
        latency_tracker = UserBotLatencyObserver()
        trace_observer = TurnTraceObserver(
            turn_tracker,
            latency_tracker=latency_tracker,
            conversation_id=conversation_id,
            tracing_context=tracing_context,
        )
        # Inject the test tracer so spans go to our in-memory exporter
        trace_observer._tracer = self._tracer
        return turn_tracker, latency_tracker, trace_observer, tracing_context

    def _all_observers(self, trace_observer):
        """Return the list of observers needed for run_test."""
        return [trace_observer._turn_tracker, trace_observer._latency_tracker, trace_observer]

    def _get_spans_by_name(self, name):
        """Return finished spans with the given name."""
        return [s for s in self._exporter.get_finished_spans() if s.name == name]

    async def test_conversation_span_created_on_start_frame(self):
        """Test that a conversation span is created when StartFrame is observed."""
        _, _, trace_observer, _ = self._create_observers(conversation_id="test-conv")
        processor = IdentityFilter()

        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=self._all_observers(trace_observer),
        )

        # End conversation to flush the conversation span (normally done by PipelineTask._cleanup)
        trace_observer.end_conversation_tracing()

        conv_spans = self._get_spans_by_name("conversation")
        self.assertEqual(len(conv_spans), 1)
        self.assertEqual(conv_spans[0].attributes["conversation.id"], "test-conv")
        self.assertEqual(conv_spans[0].attributes["conversation.type"], "voice")

    async def test_turn_spans_created_for_each_turn(self):
        """Test that a turn span is created for each conversation turn."""
        _, _, trace_observer, _ = self._create_observers()
        processor = IdentityFilter()

        frames_to_send = [
            # Turn 1
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.05),
            # Turn 2
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=self._all_observers(trace_observer),
        )

        turn_spans = self._get_spans_by_name("turn")
        self.assertEqual(len(turn_spans), 2)
        turn_numbers = {s.attributes["turn.number"] for s in turn_spans}
        self.assertEqual(turn_numbers, {1, 2})

    async def test_turn_spans_are_children_of_conversation(self):
        """Test that turn spans are parented under the conversation span."""
        _, _, trace_observer, _ = self._create_observers()
        processor = IdentityFilter()

        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=self._all_observers(trace_observer),
        )

        # End conversation to flush the conversation span
        trace_observer.end_conversation_tracing()

        conv_spans = self._get_spans_by_name("conversation")
        turn_spans = self._get_spans_by_name("turn")
        self.assertEqual(len(conv_spans), 1)
        self.assertEqual(len(turn_spans), 1)

        # Turn span's parent should be the conversation span
        conv_span_id = conv_spans[0].context.span_id
        turn_parent_id = turn_spans[0].parent.span_id
        self.assertEqual(turn_parent_id, conv_span_id)

    async def test_interrupted_turn_marked(self):
        """Test that an interrupted turn span has was_interrupted=True."""
        _, _, trace_observer, _ = self._create_observers()
        processor = IdentityFilter()

        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            # User interrupts
            UserStartedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            UserStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=self._all_observers(trace_observer),
        )

        # End conversation to flush remaining spans
        trace_observer.end_conversation_tracing()

        turn_spans = self._get_spans_by_name("turn")
        self.assertGreaterEqual(len(turn_spans), 1)
        # First turn should be interrupted
        interrupted_turns = [s for s in turn_spans if s.attributes.get("turn.was_interrupted")]
        self.assertGreaterEqual(len(interrupted_turns), 1)

    async def test_tracing_context_updated_during_turn(self):
        """Test that TracingContext is populated during a turn and cleared after."""
        tracing_ctx = TracingContext()
        _, _, trace_observer, _ = self._create_observers(tracing_context=tracing_ctx)
        processor = IdentityFilter()

        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=self._all_observers(trace_observer),
        )

        # After the turn ends, turn context should be cleared
        self.assertIsNone(tracing_ctx.get_turn_context())

    async def test_tracing_context_cleared_after_conversation_end(self):
        """Test that TracingContext is cleared when conversation tracing ends."""
        tracing_ctx = TracingContext()
        _, _, trace_observer, _ = self._create_observers(tracing_context=tracing_ctx)
        processor = IdentityFilter()

        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=self._all_observers(trace_observer),
        )

        # Manually end conversation tracing (as PipelineTask._cleanup does)
        trace_observer.end_conversation_tracing()

        self.assertIsNone(tracing_ctx.get_conversation_context())
        self.assertIsNone(tracing_ctx.get_turn_context())
        self.assertIsNone(tracing_ctx.conversation_id)

    async def test_additional_span_attributes(self):
        """Test that additional span attributes are added to the conversation span."""
        extra_attrs = {"deployment.id": "abc-123", "customer.tier": "premium"}
        tracing_ctx = TracingContext()
        turn_tracker = TurnTrackingObserver(turn_end_timeout_secs=0.2)
        latency_tracker = UserBotLatencyObserver()
        trace_observer = TurnTraceObserver(
            turn_tracker,
            latency_tracker=latency_tracker,
            additional_span_attributes=extra_attrs,
            tracing_context=tracing_ctx,
        )
        trace_observer._tracer = self._tracer
        processor = IdentityFilter()

        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[turn_tracker, latency_tracker, trace_observer],
        )

        # End conversation to flush the conversation span
        trace_observer.end_conversation_tracing()

        conv_spans = self._get_spans_by_name("conversation")
        self.assertEqual(len(conv_spans), 1)
        self.assertEqual(conv_spans[0].attributes["deployment.id"], "abc-123")
        self.assertEqual(conv_spans[0].attributes["customer.tier"], "premium")

    async def test_concurrent_pipelines_are_isolated(self):
        """Test that two pipelines with separate TracingContexts don't interfere."""
        tracing_ctx_a = TracingContext()
        tracing_ctx_b = TracingContext()

        _, _, trace_observer_a, _ = self._create_observers(
            conversation_id="conv-a", tracing_context=tracing_ctx_a
        )
        _, _, trace_observer_b, _ = self._create_observers(
            conversation_id="conv-b", tracing_context=tracing_ctx_b
        )

        processor_a = IdentityFilter()
        processor_b = IdentityFilter()

        frames = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        # Run both pipelines concurrently
        await asyncio.gather(
            run_test(
                processor_a,
                frames_to_send=frames,
                expected_down_frames=expected,
                observers=self._all_observers(trace_observer_a),
            ),
            run_test(
                processor_b,
                frames_to_send=frames,
                expected_down_frames=expected,
                observers=self._all_observers(trace_observer_b),
            ),
        )

        # End both conversations to flush spans
        trace_observer_a.end_conversation_tracing()
        trace_observer_b.end_conversation_tracing()

        # Each TracingContext should have its own conversation ID
        conv_spans = self._get_spans_by_name("conversation")
        conv_ids = {s.attributes["conversation.id"] for s in conv_spans}
        self.assertEqual(conv_ids, {"conv-a", "conv-b"})

        # Turn spans should be children of their own conversation span, not cross-linked
        turn_spans = self._get_spans_by_name("turn")
        conv_span_map = {s.context.span_id: s.attributes["conversation.id"] for s in conv_spans}
        for turn_span in turn_spans:
            parent_id = turn_span.parent.span_id
            turn_conv_id = turn_span.attributes["conversation.id"]
            parent_conv_id = conv_span_map[parent_id]
            self.assertEqual(
                turn_conv_id,
                parent_conv_id,
                f"Turn span for {turn_conv_id} parented under {parent_conv_id}",
            )

    async def test_end_conversation_closes_active_turn(self):
        """Test that end_conversation_tracing closes any active turn span."""
        _, _, trace_observer, _ = self._create_observers()

        # Manually start conversation and a turn
        trace_observer.start_conversation_tracing("conv-end-test")
        await trace_observer._handle_turn_started(1)

        self.assertIsNotNone(trace_observer._current_span)
        self.assertIsNotNone(trace_observer._conversation_span)

        # End conversation — should close both turn and conversation
        trace_observer.end_conversation_tracing()

        self.assertIsNone(trace_observer._current_span)
        self.assertIsNone(trace_observer._conversation_span)

        # Check span attributes
        turn_spans = self._get_spans_by_name("turn")
        self.assertEqual(len(turn_spans), 1)
        self.assertTrue(turn_spans[0].attributes["turn.was_interrupted"])
        self.assertTrue(turn_spans[0].attributes["turn.ended_by_conversation_end"])

    async def test_conversation_id_auto_generated(self):
        """Test that a conversation ID is auto-generated when none is provided."""
        _, _, trace_observer, _ = self._create_observers(conversation_id=None)
        processor = IdentityFilter()

        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=self._all_observers(trace_observer),
        )

        # End conversation to flush the conversation span
        trace_observer.end_conversation_tracing()

        conv_spans = self._get_spans_by_name("conversation")
        self.assertEqual(len(conv_spans), 1)
        # Should have an auto-generated UUID as conversation.id
        conv_id = conv_spans[0].attributes["conversation.id"]
        self.assertIsNotNone(conv_id)
        self.assertGreater(len(conv_id), 0)


if __name__ == "__main__":
    unittest.main()
