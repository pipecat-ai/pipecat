#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for time-to-first-byte (TTFB) metrics.

TTFB measures from a request to the first output answering it. Output that
predates the request cannot be that answer, and reporting the negative interval
it produces would put an impossible latency into a metrics stream. These tests
cover that boundary in the metrics themselves, and in the STT timeout path that
can reach it by measuring to a transcript from an earlier segment.
"""

import time
from unittest.mock import AsyncMock

import pytest

from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.services.stt_service import STTService


def _metrics() -> FrameProcessorMetrics:
    metrics = FrameProcessorMetrics()
    metrics.set_processor_name("TestProcessor")
    return metrics


class TestStopTTFBMetrics:
    """Tests for the interval TTFB reports."""

    @pytest.mark.asyncio
    async def test_reports_a_normal_interval(self):
        """A response after the request reports the time between them."""
        metrics = _metrics()
        start = time.time()
        await metrics.start_ttfb_metrics(start_time=start, report_only_initial_ttfb=False)

        frame = await metrics.stop_ttfb_metrics(end_time=start + 0.25)

        assert frame is not None
        (data,) = frame.data
        assert isinstance(data, TTFBMetricsData)
        assert data.value == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_does_not_report_output_that_predates_the_request(self):
        """Output older than the measurement is not an answer to it."""
        metrics = _metrics()
        start = time.time()
        await metrics.start_ttfb_metrics(start_time=start, report_only_initial_ttfb=False)

        assert await metrics.stop_ttfb_metrics(end_time=start - 0.3) is None

    @pytest.mark.asyncio
    async def test_a_refused_measurement_does_not_linger(self):
        """The refused measurement is closed, not left open for a later stop.

        Otherwise the next output to arrive would be measured against a start
        time that has already been rejected once.
        """
        metrics = _metrics()
        start = time.time()
        await metrics.start_ttfb_metrics(start_time=start, report_only_initial_ttfb=False)

        await metrics.stop_ttfb_metrics(end_time=start - 0.3)

        assert await metrics.stop_ttfb_metrics(end_time=start + 0.5) is None

    @pytest.mark.asyncio
    async def test_reports_nothing_when_no_measurement_is_running(self):
        """Stopping without a start is a no-op."""
        assert await _metrics().stop_ttfb_metrics() is None


class TestSTTTTFBTimeout:
    """Tests for the transcript the STT timeout path measures to."""

    @pytest.fixture
    def service(self):
        class _StubSTT(STTService):
            def can_generate_metrics(self) -> bool:
                return True

            @property
            def metrics_enabled(self) -> bool:
                # Enabled by pipeline setup, which these tests bypass. Without
                # it every measurement call is a no-op and the assertions here
                # hold for the wrong reason.
                return True

            async def run_stt(self, audio):
                if False:
                    yield audio

        service = _StubSTT()
        service._stt_ttfb_timeout = 0.0
        service.stop_ttfb_metrics = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_measures_to_the_last_transcript(self, service):
        """The transcript that arrived is what TTFB measures to."""
        service._last_transcript_time = 1000.4

        await service._ttfb_timeout_handler()

        service.stop_ttfb_metrics.assert_awaited_once_with(end_time=1000.4)

    @pytest.mark.asyncio
    async def test_a_transcript_from_before_speech_ended_reports_nothing(self, service):
        """A transcript the service finalized earlier is not this utterance's.

        The service finalized an earlier segment on its own endpointing, and no
        transcript arrived for the final segment at all. Measuring to it is
        refused where the speech end time lives, rather than compared again
        here.
        """
        service.stop_ttfb_metrics = STTService.stop_ttfb_metrics.__get__(service)
        service.push_frame = AsyncMock()
        await service.start_ttfb_metrics(start_time=1000.0)
        assert service._metrics._start_ttfb_time == 1000.0
        service._last_transcript_time = 999.6

        await service._ttfb_timeout_handler()

        service.push_frame.assert_not_awaited()
        assert service._metrics._start_ttfb_time == 0

    @pytest.mark.asyncio
    async def test_an_utterance_with_no_transcript_at_all_reports_nothing(self, service):
        """No transcript since the last VAD start means nothing to measure."""
        service._last_transcript_time = 0

        await service._ttfb_timeout_handler()

        service.stop_ttfb_metrics.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_utterance_with_no_transcript_closes_the_measurement(self, service):
        """An abandoned utterance cannot be measured against later.

        Leaving it open would let the next transcript -- which the service may
        finalize on its own before the next VAD stop -- be measured from the
        speech end of an utterance that never got an answer, reporting a large
        and entirely fictional latency.
        """
        await service.start_ttfb_metrics(start_time=1000.0)
        assert service._metrics._start_ttfb_time == 1000.0
        service._last_transcript_time = 0

        await service._ttfb_timeout_handler()

        assert service._metrics._start_ttfb_time == 0
