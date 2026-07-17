#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.utils.tracing.service_decorators import traced_gemini_live


class _StubGeminiLiveService:
    """Minimal stand-in for the Gemini Live service.

    Exposes only the attributes the ``traced_gemini_live`` decorator reads, and
    a ``_tool_result`` method decorated exactly as the real service decorates it
    — same operation, same positional signature ``(tool_call_id, tool_name,
    tool_result_message)``.
    """

    def __init__(self):
        self._tracing_enabled = True
        self._tracing_context = None
        self._settings = None
        self._model_name = "gemini-live-test"

    @traced_gemini_live(operation="llm_tool_result")
    async def _tool_result(self, tool_call_id, tool_name, tool_result_message):
        """No-op body; the decorator is what we're exercising."""
        return None


@unittest.skipUnless(HAS_OPENTELEMETRY, "opentelemetry not installed")
class TestGeminiLiveToolResultTracing(unittest.IsolatedAsyncioTestCase):
    """Tests for the ``llm_tool_result`` branch of ``traced_gemini_live``.

    Regression coverage for the decorator reading the decorated method's
    positional arguments (``tool_call_id``, ``tool_call_name``, ``result``).
    The previous implementation treated ``args[0]`` — actually the
    ``tool_call_id`` string — as a dict of result fields, so it captured
    nothing and every ``tool.*`` attribute silently went missing from the span.
    """

    @classmethod
    def setUpClass(cls):
        """Wire an in-memory span exporter into the global tracer provider.

        OpenTelemetry only allows the global tracer provider to be set once per
        process, and the decorator resolves its tracer via
        ``trace.get_tracer("pipecat")`` — so we install the provider once and
        clear the exporter between tests rather than re-setting it.
        """
        cls._exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(cls._exporter)
        existing = trace.get_tracer_provider()
        # Attach to whatever provider is already installed (another test may
        # have set one), since it can't be replaced once set; otherwise install
        # our own.
        if isinstance(existing, TracerProvider):
            existing.add_span_processor(processor)
        else:
            provider = TracerProvider()
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

    def setUp(self):
        self._exporter.clear()

    def _tool_result_span(self):
        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1, "expected exactly one llm_tool_result span")
        span = spans[0]
        self.assertEqual(span.name, "llm_tool_result")
        return span

    async def test_captures_tool_result_attributes(self):
        """The call_id, function name, and result payload land on the span.

        These are passed positionally, exactly as the real service calls
        ``_tool_result(tool_call_id, tool_name, tool_result_message)``.
        """
        service = _StubGeminiLiveService()

        await service._tool_result("call-123", "get_weather", {"temperature": 72})

        attrs = self._tool_result_span().attributes
        self.assertEqual(attrs["tool.call_id"], "call-123")
        self.assertEqual(attrs["tool.function_name"], "get_weather")
        self.assertIn('"temperature": 72', attrs["tool.result"])
        self.assertEqual(attrs["tool.result_status"], "completed")

    async def test_result_status_error(self):
        """An ``error`` key in the result dict marks the span as errored."""
        service = _StubGeminiLiveService()

        await service._tool_result("call-err", "lookup", {"error": "not found"})

        self.assertEqual(self._tool_result_span().attributes["tool.result_status"], "error")

    async def test_result_status_success(self):
        """A ``success`` key in the result dict marks the span as succeeded."""
        service = _StubGeminiLiveService()

        await service._tool_result("call-ok", "lookup", {"success": True, "value": 1})

        self.assertEqual(self._tool_result_span().attributes["tool.result_status"], "success")

    async def test_long_result_is_truncated(self):
        """Result payloads over the 2000-char limit are truncated with an ellipsis."""
        service = _StubGeminiLiveService()

        await service._tool_result("call-big", "dump", {"blob": "x" * 5000})

        result = self._tool_result_span().attributes["tool.result"]
        self.assertTrue(result.endswith("..."))
        self.assertLessEqual(len(result), 2003)

    async def test_non_dict_result_omits_result_attribute(self):
        """A non-dict result still records call_id/name but no result payload."""
        service = _StubGeminiLiveService()

        await service._tool_result("call-str", "echo", "just a string")

        attrs = self._tool_result_span().attributes
        self.assertEqual(attrs["tool.call_id"], "call-str")
        self.assertEqual(attrs["tool.function_name"], "echo")
        self.assertNotIn("tool.result", attrs)
        self.assertNotIn("tool.result_status", attrs)


if __name__ == "__main__":
    unittest.main()
