#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import threading
import unittest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

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
class TestFunctionCallSpan(unittest.IsolatedAsyncioTestCase):
    """Tests for function_call span creation in LLMService._run_function_call."""

    _provider = None
    _exporter = None

    @classmethod
    def setUpClass(cls):
        """Set up a shared TracerProvider for all tests in this class."""
        from opentelemetry.trace import set_tracer_provider

        cls._exporter = _InMemorySpanExporter()
        cls._provider = TracerProvider()
        cls._provider.add_span_processor(SimpleSpanProcessor(cls._exporter))
        set_tracer_provider(cls._provider)

    @classmethod
    def tearDownClass(cls):
        """Shut down the shared provider."""
        cls._provider.shutdown()

    def setUp(self):
        """Set up a fresh exporter and minimal LLM service for each test."""
        from unittest.mock import AsyncMock

        from opentelemetry.trace import set_span_in_context

        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.services.llm_service import (
            FunctionCallRegistryItem,
            FunctionCallRunnerItem,
            LLMService,
        )
        from pipecat.utils.tracing.tracing_context import TracingContext

        self._exporter.clear()
        self._tracer = self._provider.get_tracer("pipecat.test")

        class _StubLLMService(LLMService):
            async def run_inference(self, context, **kwargs):
                pass

            async def process_frame(self, frame, direction):
                pass

        self._service = _StubLLMService()
        self._service._tracing_enabled = True
        self._service.broadcast_frame = AsyncMock()
        self._service.broadcast_frame_instance = AsyncMock()

        self._tracing_context = TracingContext()
        turn_span = self._tracer.start_span("turn")
        turn_ctx = set_span_in_context(turn_span)
        self._tracing_context._turn_context = turn_ctx
        self._turn_span = turn_span
        self._service._tracing_context = self._tracing_context

        self._LLMContext = LLMContext
        self._FunctionCallRegistryItem = FunctionCallRegistryItem
        self._FunctionCallRunnerItem = FunctionCallRunnerItem

    def tearDown(self):
        """End the turn span."""
        self._turn_span.end()

    def _get_spans_by_name(self, name):
        """Return finished spans with the given name."""
        return [s for s in self._exporter.get_finished_spans() if s.name == name]

    async def test_function_call_span_created(self):
        """Test that a function_call span is created with correct attributes."""

        async def handler(params):
            await params.result_callback({"status": "ok"})

        self._service.register_function("test_tool", handler)

        runner_item = self._FunctionCallRunnerItem(
            registry_item=self._service._functions["test_tool"],
            function_name="test_tool",
            tool_call_id="call_123",
            arguments={"arg1": "value1"},
            context=self._LLMContext(),
        )

        await self._service._run_function_call(runner_item)

        fc_spans = self._get_spans_by_name("function_call")
        self.assertEqual(len(fc_spans), 1)
        span = fc_spans[0]
        self.assertEqual(span.attributes["tool.function_name"], "test_tool")
        self.assertEqual(span.attributes["tool.call_id"], "call_123")

        self.assertEqual(span.parent.span_id, self._turn_span.context.span_id)

    async def test_function_call_span_has_result(self):
        """Test that the function_call span captures tool.result and tool.result_status."""

        async def handler(params):
            await params.result_callback({"items": ["latte"], "total": 1})

        self._service.register_function("get_order", handler)

        runner_item = self._FunctionCallRunnerItem(
            registry_item=self._service._functions["get_order"],
            function_name="get_order",
            tool_call_id="call_456",
            arguments={},
            context=self._LLMContext(),
        )

        await self._service._run_function_call(runner_item)

        fc_spans = self._get_spans_by_name("function_call")
        self.assertEqual(len(fc_spans), 1)
        span = fc_spans[0]
        self.assertIn("tool.result", span.attributes)
        self.assertEqual(span.attributes["tool.result_status"], "success")


if __name__ == "__main__":
    unittest.main()
