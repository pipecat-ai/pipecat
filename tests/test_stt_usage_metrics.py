#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for STT usage metrics accounting and emission."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import Frame, InputAudioRawFrame, MetricsFrame, STTMetadataFrame
from pipecat.metrics.metrics import STTUsage, STTUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService, WebsocketSTTService
from pipecat.tests.utils import run_test

SAMPLE_RATE = 16000


class FakeSTTService(STTService):
    """Continuous STT service that produces no transcripts."""

    def __init__(self, **kwargs):
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        super().__init__(**kwargs)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None


class FakeWebsocketSTTService(WebsocketSTTService):
    """Websocket STT service using the default silence keepalive."""

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None

    async def _connect_websocket(self):
        pass

    async def _disconnect_websocket(self):
        pass

    async def _receive_messages(self):
        pass


def _audio_frame(seconds: float) -> InputAudioRawFrame:
    num_samples = int(SAMPLE_RATE * seconds)
    return InputAudioRawFrame(
        audio=b"\x00" * (num_samples * 2), sample_rate=SAMPLE_RATE, num_channels=1
    )


def _make_service(cls=FakeSTTService, *, usage_enabled=True):
    """Create a service ready for direct-call tests (no StartFrame)."""
    service = cls(sample_rate=SAMPLE_RATE)
    service._sample_rate = SAMPLE_RATE
    service._enable_usage_metrics = usage_enabled
    return service


@pytest.mark.asyncio
async def test_collector_returns_stt_usage_metrics_frame():
    metrics = FrameProcessorMetrics()
    metrics.set_processor_name("stt")

    frame = await metrics.start_stt_usage_metrics(STTUsage(audio_seconds=1.25))

    assert isinstance(frame, MetricsFrame)
    assert len(frame.data) == 1
    data = frame.data[0]
    assert isinstance(data, STTUsageMetricsData)
    assert data.processor == "stt"
    assert data.value.audio_seconds == 1.25


@pytest.mark.asyncio
async def test_emit_reports_accumulated_audio_and_resets():
    service = _make_service()
    pushed = []
    service.push_frame = AsyncMock(side_effect=lambda f, *args: pushed.append(f))

    service._record_stt_audio_usage(b"\x00" * (SAMPLE_RATE * 2))  # 1s
    service._record_stt_audio_usage(b"\x00" * SAMPLE_RATE)  # 0.5s
    await service.emit_stt_usage_metrics()

    assert len(pushed) == 1
    usage = pushed[0].data[0].value
    assert usage.audio_seconds == pytest.approx(1.5)

    # Pending was consumed: a second emit with no new audio is a no-op.
    await service.emit_stt_usage_metrics()
    assert len(pushed) == 1


@pytest.mark.asyncio
async def test_emit_gated_off_pushes_nothing_but_resets_pending():
    service = _make_service(usage_enabled=False)
    pushed = []
    service.push_frame = AsyncMock(side_effect=lambda f, *args: pushed.append(f))

    service._record_stt_audio_usage(b"\x00" * (SAMPLE_RATE * 2))
    await service.emit_stt_usage_metrics()

    assert pushed == []
    assert service._stt_usage_pending_seconds == 0.0


@pytest.mark.asyncio
async def test_stop_flushes_trailing_audio_usage():
    service = FakeSTTService(sample_rate=SAMPLE_RATE)

    received_down, _ = await run_test(
        service,
        frames_to_send=[_audio_frame(1.0), _audio_frame(0.5)],
        expected_down_frames=[
            STTMetadataFrame,
            InputAudioRawFrame,
            InputAudioRawFrame,
            MetricsFrame,
        ],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
    )

    metrics_frame = received_down[-1]
    data = metrics_frame.data[0]
    assert isinstance(data, STTUsageMetricsData)
    assert data.value.audio_seconds == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_usage_metrics_disabled_by_default():
    service = FakeSTTService(sample_rate=SAMPLE_RATE)

    await run_test(
        service,
        frames_to_send=[_audio_frame(1.0)],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame],
    )


@pytest.mark.asyncio
async def test_default_keepalive_silence_counts_toward_usage():
    service = _make_service(FakeWebsocketSTTService)
    service._websocket = AsyncMock()

    silence = b"\x00" * SAMPLE_RATE  # 0.5s at 16kHz mono 16-bit
    await service._send_keepalive(silence)

    assert service._stt_usage_pending_seconds == pytest.approx(0.5)


try:
    import opentelemetry  # noqa: F401

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
def test_add_stt_usage_to_span_sets_attributes():
    from pipecat.utils.tracing.service_decorators import _add_stt_usage_to_span

    span = _FakeSpan()
    _add_stt_usage_to_span(span, STTUsage(audio_seconds=1.5))
    assert span.attributes == {"metrics.audio_seconds": 1.5}


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")
@pytest.mark.asyncio
async def test_traced_stt_accumulates_pending_usage_for_span():
    # SonioxSTTService is decorated with @traced_stt, so its
    # start_stt_usage_metrics is wrapped to accumulate usage into the STT
    # span state (attached to the span when it closes).
    from pipecat.services.soniox.stt import SonioxSTTService

    service = SonioxSTTService(api_key="test-key")
    service._tracing_enabled = True

    await service.start_stt_usage_metrics(STTUsage(audio_seconds=1.0))
    await service.start_stt_usage_metrics(STTUsage(audio_seconds=0.5))

    pending = service._stt_span_state["pending_usage"]
    assert pending.audio_seconds == pytest.approx(1.5)
