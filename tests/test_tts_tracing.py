#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TTS service tracing."""

import threading
from collections.abc import AsyncGenerator

import pytest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

import pipecat.utils.tracing.service_decorators as tracing_decorators
from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSSpeakFrame, TTSStartedFrame
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test
from pipecat.utils.tracing.service_decorators import traced_tts

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


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


class _SelfStartTracedTTSService(TTSService):
    """TTS service that opens its audio context from inside ``run_tts``."""

    def __init__(self, **kwargs):
        """Initialize the test service."""
        super().__init__(
            push_start_frame=False,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        """Return whether this service can generate metrics."""
        return False

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate one synthetic audio frame."""
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            yield TTSStartedFrame(context_id=context_id)
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
@pytest.mark.asyncio
async def test_traced_tts_attaches_text_when_context_created_in_run_tts(monkeypatch):
    """TTS spans include text even when ``run_tts`` creates the audio context."""
    exporter = _InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing_decorators.trace, "get_tracer", provider.get_tracer)

    text = "delayed span text"
    try:
        await run_test(
            _SelfStartTracedTTSService(),
            enable_tracing=True,
            frames_to_send=[TTSSpeakFrame(text=text, append_to_context=False)],
        )
    finally:
        provider.shutdown()

    tts_spans = [span for span in exporter.get_finished_spans() if span.name == "tts"]
    assert len(tts_spans) == 1
    assert tts_spans[0].attributes["text"] == text
    assert tts_spans[0].attributes["metrics.character_count"] == len(text)
