#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ``metrics.ttfb`` attribute of the ``traced_tts`` and ``traced_stt`` spans.

Metrics frames of other services (such as the LLM's TTFB) pass through a service's
``push_frame`` while its span is open. The span must only take the TTFB the service itself
measured.
"""

from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from pipecat.frames.frames import Frame, MetricsFrame, TranscriptionFrame, TTSStoppedFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.tracing.service_decorators import traced_stt, traced_tts
from tests.frame_processor_helpers import frame_processor_setup

CONTEXT_ID = "ctx"


class _TestTTSService(TTSService):
    """Minimal concrete TTS service whose ``run_tts`` is traced."""

    def __init__(self, **kwargs):
        super().__init__(push_stop_frames=True, sample_rate=16000, **kwargs)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSStoppedFrame(context_id=context_id)


class _TestSTTService(STTService):
    """Minimal concrete STT service whose transcriptions are traced."""

    def __init__(self, **kwargs):
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        super().__init__(**kwargs)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None

    @traced_stt
    async def _handle_transcription(self, transcript, is_final, language=None):
        pass


def _ttfb(processor: str, value: float) -> MetricsFrame:
    return MetricsFrame(data=[TTFBMetricsData(processor=processor, value=value)])


@pytest.fixture
def spans():
    """In-memory exporter receiving the spans the tracing decorators create."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with patch(
        "pipecat.utils.tracing.service_decorators.trace.get_tracer",
        return_value=provider.get_tracer("test"),
    ):
        yield exporter


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
class TestTTSSpanTTFB:
    """The TTS span only takes the TTFB the TTS service itself measured."""

    @pytest.fixture
    def service(self):
        return _TestTTSService()

    async def _open_span(self, service: _TestTTSService) -> None:
        """Open a TTS span by creating the audio context the service is playing."""
        await service.setup(frame_processor_setup(TaskManager()))
        service._tracing_enabled = True
        await service.create_audio_context(CONTEXT_ID)
        service._audio_context_manager.playing_context_id = CONTEXT_ID

    @pytest.mark.asyncio
    async def test_ignores_the_ttfb_of_other_services(self, service, spans):
        await self._open_span(service)

        await service.push_frame(_ttfb("llm", 2.3))
        await service.push_frame(_ttfb(service.name, 0.4))
        await service.remove_audio_context(CONTEXT_ID)

        (span,) = spans.get_finished_spans()
        assert span.attributes["metrics.ttfb"] == 0.4

    @pytest.mark.asyncio
    async def test_has_no_ttfb_when_only_other_services_measured_one(self, service, spans):
        await self._open_span(service)

        await service.push_frame(_ttfb("llm", 2.3))
        await service.remove_audio_context(CONTEXT_ID)

        (span,) = spans.get_finished_spans()
        assert "metrics.ttfb" not in span.attributes


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
class TestSTTSpanTTFB:
    """The STT span only takes the TTFB the STT service itself measured."""

    @pytest.mark.asyncio
    async def test_ignores_the_ttfb_of_other_services(self, spans):
        service = _TestSTTService()
        service._tracing_enabled = True

        await service.push_frame(TranscriptionFrame("hel", "user", "now"))
        await service.push_frame(_ttfb(service.name, 0.4))
        await service.push_frame(_ttfb("llm", 2.3))
        await service.push_frame(TranscriptionFrame("hello", "user", "now", finalized=True))

        (span,) = spans.get_finished_spans()
        assert span.attributes["metrics.ttfb"] == 0.4
