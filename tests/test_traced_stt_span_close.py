#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for when ``traced_stt`` closes the ``stt`` span."""

import unittest

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.frames.frames import TranscriptionFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.tracing.service_decorators import traced_stt


class _StubSTTService:
    """Minimal stand-in for a service decorated with ``@traced_stt``.

    Exposes only what the decorator's ``push_frame`` wrapper reads. No
    ``_metrics`` attribute, so the ``UserStoppedSpeakingFrame`` path skips
    the TTFB force-stop and closes the span directly.
    """

    def __init__(self):
        self._tracing_enabled = True
        self._tracing_context = None
        self._settings = None
        self._model_name = "stt-test"
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))

    async def stop_ttfb_metrics(self, *, end_time=None):
        pass

    async def start_stt_usage_metrics(self, usage):
        pass

    @traced_stt
    async def _handle_transcription(self, transcript, is_final, language=None):
        """No-op body; the decorator is what we're exercising."""
        return None


@unittest.skipUnless(HAS_OPENTELEMETRY, "opentelemetry not installed")
class TestTracedSTTSpanClose(unittest.IsolatedAsyncioTestCase):
    """When ``traced_stt`` closes the ``stt`` span.

    A ``TranscriptionFrame`` is complete, and closes the span, when it is
    pushed with ``finalized=True`` or pushed upstream (speech-to-speech
    services such as Gemini Live and OpenAI Realtime push only complete
    transcripts upstream, with ``finalized`` left False). A downstream
    non-finalized transcript keeps the span open until a finalized one or
    the ``UserStoppedSpeakingFrame`` fallback, which tags the span
    ``stt.incomplete``.
    """

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

    def _transcript(self, text, finalized=False):
        return TranscriptionFrame(
            text=text, user_id="", timestamp="2026-01-01T00:00:00Z", finalized=finalized
        )

    async def test_upstream_transcript_closes_span_without_finalized(self):
        service = _StubSTTService()

        await service.push_frame(self._transcript("hello there"), FrameDirection.UPSTREAM)

        self.assertIsNone(service._stt_span_state["span"])
        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "stt")
        self.assertEqual(spans[0].attributes["transcript"], "hello there")
        self.assertTrue(spans[0].attributes["is_final"])
        self.assertNotIn("stt.incomplete", spans[0].attributes)

    async def test_downstream_transcript_without_finalized_keeps_span_open(self):
        service = _StubSTTService()

        await service.push_frame(self._transcript("hello"))

        self.assertIsNotNone(service._stt_span_state["span"])
        self.assertEqual(len(self._exporter.get_finished_spans()), 0)

        await service.push_frame(self._transcript("hello there", finalized=True))

        self.assertIsNone(service._stt_span_state["span"])
        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes["transcript"], "hello there")
        self.assertTrue(spans[0].attributes["is_final"])
        self.assertNotIn("stt.incomplete", spans[0].attributes)

    async def test_downstream_transcript_without_finalized_closes_on_user_stop(self):
        service = _StubSTTService()

        await service.push_frame(self._transcript("hello"))
        await service.push_frame(UserStoppedSpeakingFrame())

        self.assertIsNone(service._stt_span_state["span"])
        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertFalse(spans[0].attributes["is_final"])
        self.assertTrue(spans[0].attributes["stt.incomplete"])
