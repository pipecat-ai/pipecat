#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for time-to-first-audio (TTFA) detection and metrics."""

import time

import numpy as np
import pytest

from pipecat.audio.utils import detect_speech_onset
from pipecat.metrics.metrics import TTFAMetricsData
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics

SAMPLE_RATE = 16000


def _silence(seconds: float) -> bytes:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.int16).tobytes()


def _tone(seconds: float, amplitude: int = 5000) -> bytes:
    """A sustained tone (continuous energy, unlike an isolated blip)."""
    n = int(SAMPLE_RATE * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.int16).tobytes()


def _blip(amplitude: int = 2370) -> bytes:
    """A single sample at a realistic noise-floor amplitude."""
    return np.array([amplitude], dtype=np.int16).tobytes()


def _stereo(mono: bytes) -> bytes:
    """Duplicate a mono buffer into interleaved stereo (L == R)."""
    return np.repeat(np.frombuffer(mono, dtype=np.int16), 2).tobytes()


class TestDetectSpeechOnset:
    """Tests for the RMS-based speech onset detector."""

    def test_leading_silence_then_tone(self):
        onset = detect_speech_onset(_silence(0.5) + _tone(0.5), SAMPLE_RATE)
        assert onset is not None
        assert abs(onset / SAMPLE_RATE - 0.5) < 0.015

    def test_tone_from_start(self):
        onset = detect_speech_onset(_tone(0.2), SAMPLE_RATE)
        assert onset is not None
        assert onset / SAMPLE_RATE < 0.01

    def test_all_silence_returns_none(self):
        assert detect_speech_onset(_silence(0.3), SAMPLE_RATE) is None

    def test_empty_returns_none(self):
        assert detect_speech_onset(b"", SAMPLE_RATE) is None

    def test_too_short_to_frame_returns_none(self):
        # Fewer samples than one analysis frame: wait for more audio.
        assert detect_speech_onset(_tone(0.005), SAMPLE_RATE) is None

    def test_isolated_blip_is_rejected(self):
        # A single noise-floor blip must not be mistaken for speech onset.
        buf = _silence(0.2) + _blip() + _silence(0.2)
        assert detect_speech_onset(buf, SAMPLE_RATE) is None

    def test_brief_loud_burst_is_rejected(self):
        # A loud transient shorter than min_voiced_ms must not trigger onset.
        burst = _tone(0.01, amplitude=8000)  # 10ms < 25ms minimum voiced
        buf = _silence(0.2) + burst + _silence(0.2)
        assert detect_speech_onset(buf, SAMPLE_RATE) is None

    def test_blip_before_real_onset_locks_onto_speech(self):
        # The blip at 0.1s is ignored; onset is the sustained tone at ~0.25s.
        buf = _silence(0.1) + _blip() + _silence(0.15) + _tone(0.3)
        onset = detect_speech_onset(buf, SAMPLE_RATE)
        assert onset is not None
        assert abs(onset / SAMPLE_RATE - 0.25) < 0.02

    def test_stereo(self):
        buf = _stereo(_silence(0.1) + _tone(0.3))
        onset = detect_speech_onset(buf, SAMPLE_RATE, num_channels=2)
        assert onset is not None
        assert abs(onset / SAMPLE_RATE - 0.1) < 0.02


class TestTTFAMetrics:
    """Tests for TTFA measurement on the metrics collector."""

    def _make_metrics(self) -> FrameProcessorMetrics:
        m = FrameProcessorMetrics()
        m.set_processor_name("TestTTS")
        return m

    async def _measure_ttfb(self, m: FrameProcessorMetrics, ttfb: float):
        """Run a TTFB measurement of ``ttfb`` seconds, which arms the TTFA scan."""
        start = time.time()
        await m.start_ttfb_metrics(start_time=start, report_only_initial_ttfb=False)
        await m.stop_ttfb_metrics(end_time=start + ttfb)

    async def _process(self, m: FrameProcessorMetrics, audio: bytes, num_channels: int = 1):
        return await m.process_ttfa_metrics(
            audio=audio, sample_rate=SAMPLE_RATE, num_channels=num_channels
        )

    @pytest.mark.asyncio
    async def test_ttfa_spans_multiple_chunks(self):
        m = self._make_metrics()
        # A 0.2s TTFB measurement arms TTFA and provides its base value.
        await self._measure_ttfb(m, 0.2)

        # First chunk is entirely silence: buffered, no metric yet.
        assert await self._process(m, _silence(0.3)) is None

        # Second chunk completes the leading silence (0.4s total) then tone.
        frame = await self._process(m, _silence(0.1) + _tone(0.3))
        assert frame is not None
        data = frame.data[0]
        assert isinstance(data, TTFAMetricsData)
        # 0.2s TTFB + 0.4s leading silence, carried as a breakdown.
        assert data.ttfa == pytest.approx(0.6, abs=0.02)
        assert data.ttfb == pytest.approx(0.2, abs=1e-3)
        assert data.leading_silence == pytest.approx(0.4, abs=0.02)
        assert data.ttfa == pytest.approx(data.ttfb + data.leading_silence)

    @pytest.mark.asyncio
    async def test_ttfa_reported_once_per_response(self):
        m = self._make_metrics()
        await self._measure_ttfb(m, 0.1)

        first = await self._process(m, _tone(0.2))
        assert first is not None
        # Subsequent audio in the same response does not re-report.
        assert await self._process(m, _tone(0.2)) is None

    @pytest.mark.asyncio
    async def test_no_metric_without_ttfb(self):
        m = self._make_metrics()
        # TTFB never measured, so the scan is never armed: processing is a no-op.
        assert await self._process(m, _tone(0.2)) is None

    @pytest.mark.asyncio
    async def test_all_silence_response_emits_nothing(self):
        m = self._make_metrics()
        await self._measure_ttfb(m, 0.1)
        assert await self._process(m, _silence(0.5)) is None

    @pytest.mark.asyncio
    async def test_ttfa_rearmed_each_response(self):
        m = self._make_metrics()

        # First response: 0.1s TTFB, no leading silence.
        await self._measure_ttfb(m, 0.1)
        first = await self._process(m, _tone(0.2))
        assert first is not None
        assert first.data[0].ttfa == pytest.approx(0.1, abs=0.02)

        # Second response re-arms via a fresh TTFB stop; buffer/silence reset.
        await self._measure_ttfb(m, 0.3)
        second = await self._process(m, _silence(0.2) + _tone(0.2))
        assert second is not None
        assert second.data[0].ttfa == pytest.approx(0.5, abs=0.02)
