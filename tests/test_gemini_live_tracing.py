#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ``llm_setup`` span of ``traced_gemini_live``.

``llm_setup`` is the one operation left on this decorator: replies and tool
calls are traced by ``traced_realtime_llm`` (see ``test_traced_realtime_llm``).
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from google.genai.types import FunctionDeclaration, Tool

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import CancelFrame, StartFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.utils.asyncio.task_manager import TaskManager


class _FakeSession:
    def __init__(self):
        self._closed = asyncio.Event()

    async def _messages(self):
        await self._closed.wait()
        return
        yield

    def receive(self):
        return self._messages()

    async def close(self):
        self._closed.set()


class _FakeConnection:
    async def __aenter__(self):
        return _FakeSession()

    async def __aexit__(self, *exc):
        return False


def _function(name: str) -> FunctionSchema:
    return FunctionSchema(name=name, description=f"{name} tool", properties={}, required=[])


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

    def _setup_spans(self):
        return [s for s in self._exporter.get_finished_spans() if s.name == "llm_setup"]

    async def _wait_for_setup_spans(self, count):
        for _ in range(200):
            if len(self._setup_spans()) >= count:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"expected {count} llm_setup spans, got {len(self._setup_spans())}")

    async def _run(self, service, context=None):
        with (
            patch.object(service._client.aio.live, "connect", return_value=_FakeConnection()),
            patch.object(service, "_create_initial_response", new=AsyncMock()),
        ):
            await service.setup(
                FrameProcessorSetup(
                    clock=SystemClock(),
                    task_manager=TaskManager(),
                    pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
                    enable_tracing=True,
                )
            )
            await service.start(StartFrame())
            try:
                await self._wait_for_setup_spans(1)
                if context is not None:
                    await service._handle_context(context)
                    await self._wait_for_setup_spans(2)
            finally:
                await service.cancel(CancelFrame())
        return self._setup_spans()

    async def test_records_tools_and_system_instruction(self):
        service = GeminiLiveLLMService(
            api_key="test-key",
            settings=GeminiLiveLLMService.Settings(
                model="gemini-live-test", system_instruction="Be brief."
            ),
            tools=ToolsSchema(
                standard_tools=[_function("get_weather"), _function("get_time")],
                custom_tools={AdapterType.GEMINI: [{"google_search": {}}]},
            ),
        )

        (span,) = await self._run(service)

        attrs = span.attributes
        self.assertEqual(attrs["gen_ai.operation.name"], "llm_setup")
        self.assertEqual(attrs["gen_ai.request.model"], "gemini-live-test")
        self.assertEqual(attrs["tools.count"], 2)
        self.assertEqual(attrs["tools.names"], "get_weather,get_time")
        definitions = json.loads(attrs["tools.definitions"])
        self.assertEqual([d["name"] for d in definitions], ["get_weather", "get_time"])
        self.assertEqual(attrs["gen_ai.system_instructions"], "Be brief.")

    async def test_records_the_tools_of_each_connect(self):
        """Constructor tools are provider-native ``Tool`` objects, not the adapter's dicts."""
        service = GeminiLiveLLMService(
            api_key="test-key",
            tools=[Tool(function_declarations=[FunctionDeclaration(name="get_weather")])],
        )
        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
            tools=ToolsSchema(standard_tools=[_function("get_time")]),
        )

        first, reconnect = await self._run(service, context)

        self.assertEqual(first.attributes["tools.names"], "get_weather")
        self.assertEqual(
            json.loads(first.attributes["tools.definitions"])[0]["name"], "get_weather"
        )
        self.assertEqual(reconnect.attributes["tools.names"], "get_time")

    async def test_tracing_disabled_records_nothing(self):
        service = GeminiLiveLLMService(api_key="test-key")

        await service._handle_session_ready(_FakeSession())

        self.assertEqual(self._exporter.get_finished_spans(), ())
