#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ``llm_setup`` span of ``traced_gemini_live``.

``llm_setup`` is the one operation left on this decorator: replies and tool
calls are traced by ``traced_realtime_llm`` (see ``test_traced_realtime_llm``).
"""

import unittest

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.utils.tracing.service_decorators import traced_gemini_live


class _StubSettings:
    def __init__(self, system_instruction):
        self.system_instruction = system_instruction

    def given_fields(self):
        return {}


class _StubGeminiLiveService:
    """Minimal stand-in exposing only what ``traced_gemini_live`` reads."""

    def __init__(self, tools=None, system_instruction=None):
        self._tracing_enabled = True
        self._tracing_context = None
        self._settings = None
        self.settings = _StubSettings(system_instruction)
        self._model_name = "gemini-live-test"
        self._tools = tools
        self._context = None

    @traced_gemini_live(operation="llm_setup")
    async def _setup(self):
        return None


@unittest.skipUnless(HAS_OPENTELEMETRY, "opentelemetry not installed")
class TestGeminiLiveSetupTracing(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls._exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(cls._exporter)
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            existing.add_span_processor(processor)
        else:
            provider = TracerProvider()
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

    def setUp(self):
        self._exporter.clear()

    def _setup_span(self):
        spans = [s for s in self._exporter.get_finished_spans() if s.name == "llm_setup"]
        self.assertEqual(len(spans), 1)
        return spans[0]

    async def test_records_tools_and_system_instruction(self):
        tools = ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="get_weather",
                    description="Weather lookup",
                    properties={"city": {"type": "string"}},
                    required=["city"],
                )
            ]
        )
        service = _StubGeminiLiveService(tools=tools, system_instruction="Be brief.")

        await service._setup()

        attrs = self._setup_span().attributes
        self.assertEqual(attrs["gen_ai.operation.name"], "llm_setup")
        self.assertEqual(attrs["gen_ai.request.model"], "gemini-live-test")
        self.assertEqual(attrs["tools.names"], "get_weather")
        self.assertEqual(attrs["gen_ai.system_instructions"], "Be brief.")

    async def test_tracing_disabled_records_nothing(self):
        service = _StubGeminiLiveService()
        service._tracing_enabled = False

        await service._setup()

        self.assertEqual(self._exporter.get_finished_spans(), ())
