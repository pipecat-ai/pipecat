#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import threading
import unittest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.utils.tracing.service_attributes import (
    add_stt_span_attributes,
    add_tts_span_attributes,
)

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
class TestTracingServiceAttributes(unittest.TestCase):
    """Tests for TTS and STT service tracing attributes."""

    def setUp(self):
        """Set up a fresh provider and exporter for each test."""
        self._exporter = _InMemorySpanExporter()
        self._provider = TracerProvider()
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._tracer = self._provider.get_tracer("test")

    def tearDown(self):
        """Shut down the provider to flush spans."""
        self._provider.shutdown()

    def test_tts_span_with_audio_data(self):
        """Test that TTS span attributes are set correctly when audio data is provided."""
        span = self._tracer.start_span("tts_test")
        audio = b"dummy_audio_bytes"
        add_tts_span_attributes(
            span=span,
            service_name="elevenlabs",
            model="eleven_monolingual_v1",
            voice_id="voice-id-123",
            audio_data=audio,
        )
        span.end()

        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        target_span = spans[0]

        # audio.data_size_bytes is set correctly
        self.assertEqual(target_span.attributes.get("audio.data_size_bytes"), len(audio))

        # output is set to base64 encoded audio string
        expected_base64 = base64.b64encode(audio).decode("utf-8")
        self.assertEqual(target_span.attributes.get("output"), expected_base64)

        # langfuse.media is valid base64 JSON with type: audio
        media_json = target_span.attributes.get("langfuse.media")
        self.assertIsNotNone(media_json)
        media_data = json.loads(media_json)
        self.assertEqual(media_data.get("type"), "audio")
        self.assertEqual(media_data.get("data"), expected_base64)
        self.assertEqual(media_data.get("mediaType"), "audio/wav")

    def test_stt_span_with_audio_data(self):
        """Test that STT span attributes are set correctly when audio data is provided."""
        span = self._tracer.start_span("stt_test")
        audio = b"dummy_audio_bytes"
        add_stt_span_attributes(
            span=span,
            service_name="deepgram",
            model="nova-2",
            audio_data=audio,
        )
        span.end()

        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        target_span = spans[0]

        # audio.data_size_bytes is set correctly
        self.assertEqual(target_span.attributes.get("audio.data_size_bytes"), len(audio))

        # input is set to base64 encoded audio string
        expected_base64 = base64.b64encode(audio).decode("utf-8")
        self.assertEqual(target_span.attributes.get("input"), expected_base64)

        # langfuse.media is valid base64 JSON
        media_json = target_span.attributes.get("langfuse.media")
        self.assertIsNotNone(media_json)
        media_data = json.loads(media_json)
        self.assertEqual(media_data.get("type"), "audio")
        self.assertEqual(media_data.get("data"), expected_base64)
        self.assertEqual(media_data.get("mediaType"), "audio/wav")

    def test_tts_span_with_metadata(self):
        """Test that custom metadata is flattened and set as span attributes."""
        span = self._tracer.start_span("tts_test")
        metadata = {"session_type": "web", "user_tier": "premium"}
        add_tts_span_attributes(
            span=span,
            service_name="elevenlabs",
            model="eleven_monolingual_v1",
            voice_id="voice-id-123",
            metadata=metadata,
        )
        span.end()

        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        target_span = spans[0]

        self.assertEqual(target_span.attributes.get("metadata.session_type"), "web")
        self.assertEqual(target_span.attributes.get("metadata.user_tier"), "premium")

    def test_stt_span_with_metadata(self):
        """Test that custom metadata is flattened and set as span attributes for STT."""
        span = self._tracer.start_span("stt_test")
        metadata = {"session_type": "web", "user_tier": "premium"}
        add_stt_span_attributes(
            span=span,
            service_name="deepgram",
            model="nova-2",
            metadata=metadata,
        )
        span.end()

        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        target_span = spans[0]

        self.assertEqual(target_span.attributes.get("metadata.session_type"), "web")
        self.assertEqual(target_span.attributes.get("metadata.user_tier"), "premium")

    def test_metadata_skips_non_primitives(self):
        """Test that metadata skips non-primitive nested values without throwing errors."""
        span = self._tracer.start_span("metadata_test")
        metadata = {"nested": {"a": 1}, "primitive": "allowed"}
        add_tts_span_attributes(
            span=span,
            service_name="elevenlabs",
            model="eleven_monolingual_v1",
            voice_id="voice-id-123",
            metadata=metadata,
        )
        span.end()

        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        target_span = spans[0]

        self.assertNotIn("metadata.nested", target_span.attributes)
        self.assertEqual(target_span.attributes.get("metadata.primitive"), "allowed")

    def test_no_audio_no_media_attribute(self):
        """Test that attributes related to audio are not set if audio data is not provided."""
        span = self._tracer.start_span("no_audio_test")
        add_tts_span_attributes(
            span=span,
            service_name="elevenlabs",
            model="eleven_monolingual_v1",
            voice_id="voice-id-123",
        )
        span.end()

        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        target_span = spans[0]

        self.assertNotIn("audio.data_size_bytes", target_span.attributes)
        self.assertNotIn("langfuse.media", target_span.attributes)
        self.assertNotIn("output", target_span.attributes)

    def test_existing_tts_attributes_unchanged(self):
        """Test that existing attributes still behave the same as before."""
        span = self._tracer.start_span("existing_test")
        add_tts_span_attributes(
            span=span,
            service_name="elevenlabs",
            model="eleven_monolingual_v1",
            voice_id="voice-id-123",
            text="hello world",
            character_count=11,
            ttfb=0.5,
        )
        span.end()

        spans = self._exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        target_span = spans[0]

        self.assertEqual(target_span.attributes.get("gen_ai.provider.name"), "elevenlabs")
        self.assertEqual(
            target_span.attributes.get("gen_ai.request.model"), "eleven_monolingual_v1"
        )
        self.assertEqual(target_span.attributes.get("voice_id"), "voice-id-123")
        self.assertEqual(target_span.attributes.get("text"), "hello world")
        self.assertEqual(target_span.attributes.get("metrics.character_count"), 11)
        self.assertEqual(target_span.attributes.get("metrics.ttfb"), 0.5)


if __name__ == "__main__":
    unittest.main()
