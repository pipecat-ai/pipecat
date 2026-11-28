#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ``llm_setup`` span of ``traced_openai_realtime``."""

import unittest

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.zoom_ai.realtime.llm import ZoomAIRealtimeLLMService


def _tool(name):
    return FunctionSchema(name=name, description=name, properties={}, required=[])


def _service(cls=OpenAIRealtimeLLMService):
    return cls(
        api_key="test-key",
        settings=cls.Settings(
            system_instruction="Be brief.",
            session_properties=events.SessionProperties(
                tools=ToolsSchema(standard_tools=[_tool("get_weather")])
            ),
        ),
    )


async def _create_session(service):
    service._tracing_enabled = True
    await service._handle_evt_session_created(
        events.SessionCreatedEvent(
            event_id="evt_1", type="session.created", session=events.SessionProperties()
        )
    )


@unittest.skipUnless(HAS_OPENTELEMETRY, "opentelemetry not installed")
class TestOpenAIRealtimeSetupTracing(unittest.IsolatedAsyncioTestCase):
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

    async def test_records_session_instructions_and_tools(self):
        await _create_session(_service())

        span = self._setup_span()
        self.assertEqual(span.status.status_code, StatusCode.UNSET)
        self.assertEqual(span.attributes["instructions"], "Be brief.")
        self.assertEqual(span.attributes["session.instructions"], "Be brief.")
        self.assertEqual(span.attributes["tools.names"], "get_weather")

    async def test_records_the_context_tools_that_were_sent(self):
        service = _service()
        service._context = LLMContext(tools=ToolsSchema(standard_tools=[_tool("get_news")]))

        await _create_session(service)

        attrs = self._setup_span().attributes
        self.assertEqual(attrs["instructions"], "Be brief.")
        self.assertEqual(attrs["tools.names"], "get_news")

    async def test_zoom_records_its_session_instructions_and_tools(self):
        await _create_session(_service(ZoomAIRealtimeLLMService))

        attrs = self._setup_span().attributes
        self.assertEqual(attrs["instructions"], "Be brief.")
        self.assertEqual(attrs["tools.names"], "get_weather")
