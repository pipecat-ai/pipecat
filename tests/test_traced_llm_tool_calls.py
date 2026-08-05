#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.frames.frames import FunctionCallFromLLM, FunctionCallsStartedFrame, LLMTextFrame
from pipecat.utils.tracing.service_decorators import traced_llm


class _StubLLMService:
    """Minimal stand-in exposing only what ``traced_llm`` reads."""

    def __init__(self, frames):
        self._tracing_enabled = True
        self._frames = frames
        self._model_name = "test-model"

    async def push_frame(self, frame, direction=None):
        return None

    @traced_llm
    async def _process_context(self, context):
        for frame in self._frames:
            await self.push_frame(frame)


def _make_call(name, tool_call_id, arguments):
    return FunctionCallFromLLM(
        function_name=name, tool_call_id=tool_call_id, arguments=arguments, context=None
    )


@unittest.skipUnless(HAS_OPENTELEMETRY, "opentelemetry not installed")
class TestTracedLLMToolCallOutput(unittest.IsolatedAsyncioTestCase):
    """The LLM span records function calls, not just LLMTextFrame text."""

    @classmethod
    def setUpClass(cls):
        cls._exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(cls._exporter)
        existing = trace.get_tracer_provider()
        # The global provider can only be set once per process; attach to an
        # existing SDK provider if another test already installed one.
        if isinstance(existing, TracerProvider):
            existing.add_span_processor(processor)
        else:
            provider = TracerProvider()
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

    def setUp(self):
        self._exporter.clear()

    def _llm_span(self):
        spans = [s for s in self._exporter.get_finished_spans() if s.name == "llm"]
        self.assertEqual(len(spans), 1)
        return spans[0]

    async def test_tool_only_response_records_output(self):
        calls = [_make_call("get_weather", "call_1", {"city": "Berlin"})]
        service = _StubLLMService([FunctionCallsStartedFrame(function_calls=calls)])
        await service._process_context(context=None)

        span = self._llm_span()
        expected = [{"name": "get_weather", "arguments": {"city": "Berlin"}}]
        self.assertEqual(json.loads(span.attributes["tool_calls"]), expected)
        self.assertEqual(json.loads(span.attributes["output"]), expected)

    async def test_mixed_response_records_text_and_tool_calls(self):
        calls = [_make_call("get_weather", "call_1", {"city": "Berlin"})]
        service = _StubLLMService(
            [
                LLMTextFrame(text="Let me check."),
                FunctionCallsStartedFrame(function_calls=calls),
            ]
        )
        await service._process_context(context=None)

        span = self._llm_span()
        self.assertEqual(span.attributes["output"], "Let me check.")
        self.assertEqual(
            json.loads(span.attributes["tool_calls"]),
            [{"name": "get_weather", "arguments": {"city": "Berlin"}}],
        )

    async def test_broadcast_duplicates_are_deduped(self):
        calls = [_make_call("get_weather", "call_1", {"city": "Berlin"})]
        # broadcast_frame pushes the frame both downstream and upstream.
        frames = [
            FunctionCallsStartedFrame(function_calls=calls),
            FunctionCallsStartedFrame(function_calls=calls),
        ]
        service = _StubLLMService(frames)
        await service._process_context(context=None)

        span = self._llm_span()
        self.assertEqual(len(json.loads(span.attributes["tool_calls"])), 1)
