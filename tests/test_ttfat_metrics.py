#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for time-to-first-answer-token (TTFAT) metrics."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentResponse,
    Part,
)

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import TTFATMetricsData, TTFBMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def _google_chunk(text: str) -> GenerateContentResponse:
    """A streamed chunk carrying model output."""
    return GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[Part(text=text)]))]
    )


def _google_chunk_with_function_call(name: str) -> GenerateContentResponse:
    """A streamed chunk carrying a tool call."""
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    role="model",
                    parts=[Part(function_call=FunctionCall(name=name, args={}))],
                )
            )
        ]
    )


class TestTTFATMetrics:
    """TTFAT measurement in the metrics collector."""

    def _make_metrics(self) -> FrameProcessorMetrics:
        m = FrameProcessorMetrics()
        m.set_processor_name("TestLLM")
        return m

    async def _measure_ttfb(
        self,
        m: FrameProcessorMetrics,
        start: float,
        ttfb: float,
        *,
        report_only_initial_ttfb: bool = False,
    ):
        """Run a TTFB measurement of ``ttfb`` seconds, which arms TTFAT."""
        await m.start_ttfb_metrics(
            start_time=start, report_only_initial_ttfb=report_only_initial_ttfb
        )
        await m.stop_ttfb_metrics(end_time=start + ttfb)

    @pytest.mark.asyncio
    async def test_ttfat_carries_breakdown(self):
        m = self._make_metrics()
        start = time.time()
        await self._measure_ttfb(m, start, 0.2)

        frame = await m.stop_ttfat_metrics(end_time=start + 0.9)
        assert frame is not None
        data = frame.data[0]
        assert isinstance(data, TTFATMetricsData)
        assert data.ttfat == pytest.approx(0.9, abs=1e-3)
        assert data.ttfb == pytest.approx(0.2, abs=1e-3)
        assert data.thinking_time == pytest.approx(0.7, abs=1e-3)
        assert data.ttfat == pytest.approx(data.ttfb + data.thinking_time)

    @pytest.mark.asyncio
    async def test_ttfat_measures_from_the_request_not_from_ttfb(self):
        """The two measurements share a start, so TTFAT is never the gap alone."""
        m = self._make_metrics()
        start = time.time()
        await self._measure_ttfb(m, start, 0.5)

        frame = await m.stop_ttfat_metrics(end_time=start + 0.6)
        assert frame is not None
        assert frame.data[0].ttfat == pytest.approx(0.6, abs=1e-3)
        assert frame.data[0].thinking_time == pytest.approx(0.1, abs=1e-3)

    @pytest.mark.asyncio
    async def test_answer_token_arriving_with_first_output_reports_no_thinking(self):
        m = self._make_metrics()
        start = time.time()
        await self._measure_ttfb(m, start, 0.3)

        frame = await m.stop_ttfat_metrics(end_time=start + 0.3)
        assert frame is not None
        assert frame.data[0].thinking_time == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_ttfat_reported_once_per_response(self):
        m = self._make_metrics()
        start = time.time()
        await self._measure_ttfb(m, start, 0.1)

        assert await m.stop_ttfat_metrics(end_time=start + 0.4) is not None
        # Later tokens in the same response do not re-report.
        assert await m.stop_ttfat_metrics(end_time=start + 0.5) is None

    @pytest.mark.asyncio
    async def test_no_metric_without_ttfb(self):
        """Nothing arms TTFAT if the response produced no output at all."""
        m = self._make_metrics()
        assert await m.stop_ttfat_metrics(end_time=time.time()) is None

    @pytest.mark.asyncio
    async def test_interrupted_response_does_not_leak_into_the_next(self):
        """A response cut off before its answer token reports nothing."""
        m = self._make_metrics()
        first = time.time()
        await self._measure_ttfb(m, first, 0.2)
        # No answer token arrives; the next request starts instead.
        second = first + 10.0
        await self._measure_ttfb(m, second, 0.3)

        frame = await m.stop_ttfat_metrics(end_time=second + 0.8)
        assert frame is not None
        # Measured against the second request, not the abandoned first.
        assert frame.data[0].ttfat == pytest.approx(0.8, abs=1e-3)
        assert frame.data[0].thinking_time == pytest.approx(0.5, abs=1e-3)

    @pytest.mark.asyncio
    async def test_interrupted_response_does_not_leak_when_reporting_only_initial_ttfb(self):
        """``report_only_initial_ttfb`` stops later requests measuring anything at all."""
        m = self._make_metrics()
        first = time.time()
        await self._measure_ttfb(m, first, 0.2, report_only_initial_ttfb=True)
        # No answer token arrives; the next request starts instead. Its TTFB
        # goes unmeasured, so there is no request start to measure TTFAT from.
        second = first + 10.0
        await self._measure_ttfb(m, second, 0.3, report_only_initial_ttfb=True)

        assert await m.stop_ttfat_metrics(end_time=second + 0.8) is None


class TestTTFATServiceReporting:
    """Which services report TTFAT, and when they record it."""

    def test_text_service_reports_ttfat(self):
        assert GoogleLLMService(api_key="test-key").reports_ttfat is True

    def test_speech_to_speech_service_does_not_report_ttfat(self):
        assert GeminiLiveLLMService(api_key="test-key").reports_ttfat is False

    @pytest.mark.asyncio
    async def test_recorded_even_when_turn_completion_withholds_the_text(self):
        """The measurement belongs to the model, not to what the pipeline does next.

        Turn-completion filtering can hold a response's text back or drop it, so
        recording TTFAT downstream of that would time the filter instead.
        """
        service = GoogleLLMService(api_key="test-key")
        service._filter_incomplete_user_turns = True
        # State in which _push_turn_text drops the text outright.
        service._user_turn_completion_voiced = True
        service._turn_marker = None

        recorded = []
        pushed = []

        async def fake_stop_ttfat(**kwargs):
            recorded.append(True)

        async def capture_frame(frame, direction=None):
            pushed.append(frame)

        with (
            patch.object(service, "stop_ttfat_metrics", fake_stop_ttfat),
            patch.object(service, "push_frame", capture_frame),
        ):
            await service._push_llm_text("Hello")

        assert recorded, "TTFAT should be recorded when the model produces the token"
        assert not pushed, "no text reaches the pipeline in this state"

    @pytest.mark.asyncio
    async def test_streamed_response_pushes_a_ttfat_frame(self):
        """The full path: a streamed response emits TTFAT alongside TTFB."""
        service = GoogleLLMService(api_key="test-key")
        await service.setup(frame_processor_setup(TaskManager(), enable_metrics=True))

        frames = []

        async def capture_frame(frame, direction=None):
            frames.append(frame)

        async def fake_stream(context):
            async def generator():
                yield _google_chunk("Hello")

            return generator()

        with (
            patch.object(service, "push_frame", capture_frame),
            patch.object(service, "_stream_content", fake_stream),
        ):
            await service._process_context(LLMContext())

        metrics = [d for f in frames if isinstance(f, MetricsFrame) for d in f.data]
        ttfat = [d for d in metrics if isinstance(d, TTFATMetricsData)]
        ttfb = [d for d in metrics if isinstance(d, TTFBMetricsData)]
        assert len(ttfb) == 1, "TTFB is still reported on its own"
        assert len(ttfat) == 1, "TTFAT is reported once for the response"
        assert ttfat[0].ttfat >= ttfat[0].ttfb
        assert ttfat[0].thinking_time == pytest.approx(ttfat[0].ttfat - ttfat[0].ttfb)


class TestTTFATToolCalls:
    """A tool-only turn ends TTFAT at its first tool call, not at end of stream."""

    async def _stop_index(self, service, patch_stream, chunks) -> int | None:
        """Stream canned chunks and report which one ended TTFAT."""
        state = {"index": -1, "stopped_at": None}

        async def fake_stop_ttfat(**kwargs):
            if state["stopped_at"] is None:
                state["stopped_at"] = state["index"]

        async def generator():
            for index, chunk in enumerate(chunks):
                state["index"] = index
                yield chunk

        async def capture_frame(frame, direction=None):
            pass

        with (
            patch.object(service, "push_frame", capture_frame),
            patch.object(service, "stop_ttfat_metrics", fake_stop_ttfat),
            patch.object(service, "run_function_calls", AsyncMock()),
            patch_stream(service, generator),
        ):
            await service._process_context(LLMContext())

        return state["stopped_at"]

    @pytest.mark.asyncio
    async def test_google_stops_at_the_tool_call_not_end_of_stream(self):
        service = GoogleLLMService(api_key="test-key")

        def patch_stream(svc, generator):
            async def fake_stream(context):
                return generator()

            return patch.object(svc, "_stream_content", fake_stream)

        chunks = [
            _google_chunk_with_function_call("get_weather"),
            # Trailing chunks the model streams before the response ends.
            _google_chunk(""),
            _google_chunk(""),
        ]
        assert await self._stop_index(service, patch_stream, chunks) == 0

    @pytest.mark.asyncio
    async def test_anthropic_stops_at_the_tool_call_not_end_of_stream(self):
        service = AnthropicLLMService(api_key="test-key")

        def patch_stream(svc, generator):
            async def fake_stream(api_call, params):
                return generator()

            return patch.object(svc, "_create_message_stream", fake_stream)

        events = [
            SimpleNamespace(type="message_start"),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", id="t1", name="get_weather"),
            ),
            # Arguments stream after the call begins; TTFAT must not wait for them.
            SimpleNamespace(
                type="content_block_delta", delta=SimpleNamespace(partial_json='{"city":')
            ),
            SimpleNamespace(
                type="content_block_delta", delta=SimpleNamespace(partial_json='"Paris"}')
            ),
        ]
        assert await self._stop_index(service, patch_stream, events) == 1

    @pytest.mark.asyncio
    async def test_answer_text_before_a_tool_call_wins(self):
        """Text and tool calls share one measurement; whichever comes first ends it."""
        service = GoogleLLMService(api_key="test-key")

        def patch_stream(svc, generator):
            async def fake_stream(context):
                return generator()

            return patch.object(svc, "_stream_content", fake_stream)

        chunks = [
            _google_chunk("Let me look that up."),
            _google_chunk_with_function_call("get_weather"),
        ]
        assert await self._stop_index(service, patch_stream, chunks) == 0
