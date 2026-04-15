#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

try:
    from opentelemetry.sdk.trace import TracerProvider

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.utils.tracing.tracing_context import TracingContext


@unittest.skipUnless(HAS_OPENTELEMETRY, "opentelemetry not installed")
class TestTracingContext(unittest.TestCase):
    """Tests for TracingContext."""

    @classmethod
    def setUpClass(cls):
        """Set up a tracer provider for generating span contexts."""
        cls._provider = TracerProvider()
        cls._tracer = cls._provider.get_tracer("test")

    def test_initial_state_is_empty(self):
        """Test that a new TracingContext starts with no context set."""
        ctx = TracingContext()
        self.assertIsNone(ctx.get_conversation_context())
        self.assertIsNone(ctx.get_turn_context())
        self.assertIsNone(ctx.conversation_id)

    def test_set_and_get_conversation_context(self):
        """Test setting and retrieving conversation context."""
        ctx = TracingContext()
        span = self._tracer.start_span("conv")
        span_context = span.get_span_context()

        ctx.set_conversation_context(span_context, "conv-123")

        self.assertIsNotNone(ctx.get_conversation_context())
        self.assertEqual(ctx.conversation_id, "conv-123")
        span.end()

    def test_clear_conversation_context(self):
        """Test clearing conversation context by passing None."""
        ctx = TracingContext()
        span = self._tracer.start_span("conv")

        ctx.set_conversation_context(span.get_span_context(), "conv-123")
        self.assertIsNotNone(ctx.get_conversation_context())

        ctx.set_conversation_context(None)
        self.assertIsNone(ctx.get_conversation_context())
        self.assertIsNone(ctx.conversation_id)
        span.end()

    def test_set_and_get_turn_context(self):
        """Test setting and retrieving turn context."""
        ctx = TracingContext()
        span = self._tracer.start_span("turn")
        span_context = span.get_span_context()

        ctx.set_turn_context(span_context)

        self.assertIsNotNone(ctx.get_turn_context())
        span.end()

    def test_clear_turn_context(self):
        """Test clearing turn context by passing None."""
        ctx = TracingContext()
        span = self._tracer.start_span("turn")

        ctx.set_turn_context(span.get_span_context())
        self.assertIsNotNone(ctx.get_turn_context())

        ctx.set_turn_context(None)
        self.assertIsNone(ctx.get_turn_context())
        span.end()

    def test_generate_conversation_id(self):
        """Test that generated conversation IDs are unique UUIDs."""
        id1 = TracingContext.generate_conversation_id()
        id2 = TracingContext.generate_conversation_id()
        self.assertIsInstance(id1, str)
        self.assertNotEqual(id1, id2)

    def test_instances_are_isolated(self):
        """Test that two TracingContext instances do not share state."""
        ctx_a = TracingContext()
        ctx_b = TracingContext()

        span = self._tracer.start_span("turn")

        ctx_a.set_turn_context(span.get_span_context())
        ctx_a.set_conversation_context(span.get_span_context(), "conv-a")

        # ctx_b should still be empty
        self.assertIsNone(ctx_b.get_turn_context())
        self.assertIsNone(ctx_b.get_conversation_context())
        self.assertIsNone(ctx_b.conversation_id)
        span.end()

    def test_conversation_and_turn_are_independent(self):
        """Test that clearing turn context does not affect conversation context."""
        ctx = TracingContext()
        conv_span = self._tracer.start_span("conv")
        turn_span = self._tracer.start_span("turn")

        ctx.set_conversation_context(conv_span.get_span_context(), "conv-1")
        ctx.set_turn_context(turn_span.get_span_context())

        # Clear turn but conversation should remain
        ctx.set_turn_context(None)
        self.assertIsNone(ctx.get_turn_context())
        self.assertIsNotNone(ctx.get_conversation_context())
        self.assertEqual(ctx.conversation_id, "conv-1")

        conv_span.end()
        turn_span.end()


if __name__ == "__main__":
    unittest.main()
