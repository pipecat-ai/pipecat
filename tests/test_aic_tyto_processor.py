#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import threading
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

aic_sdk: Any
try:
    import aic_sdk

    HAS_AIC_SDK = True
except ImportError:
    aic_sdk = None
    HAS_AIC_SDK = False

from pipecat.frames.frames import (
    AudioQualityFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import (
    AudioQualityMetricsData,
    AudioQualityScores,
    AudioQualitySummaryMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from tests.aic_mocks import MockAnalysisResult, MockAnalyzer, MockCollector, MockModel

AIC_TYTO_MODULE = "pipecat.processors.audio.aic_tyto_processor"

SAMPLE_RATE = 16000


def _audio_frame(secs: float) -> InputAudioRawFrame:
    samples = np.zeros(int(secs * SAMPLE_RATE), dtype=np.int16)
    return InputAudioRawFrame(audio=samples.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1)


class _CancelProbe(FrameProcessor):
    """Records frames and releases ``gate`` while a CancelFrame push is in progress."""

    def __init__(self, gate: threading.Event):
        super().__init__(enable_direct_mode=True)
        self.gate = gate
        self.frames = []

    async def process_frame(self, frame, direction: FrameDirection):
        self.frames.append(frame)
        if isinstance(frame, CancelFrame):
            self.gate.set()
            await asyncio.sleep(0.1)
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _TapFilter:
    def __init__(self):
        self.tap = None
        self.taps = []

    def set_pre_process_tap(self, tap):
        self.tap = tap
        self.taps.append(tap)


class _TapFeeder(FrameProcessor):
    """Stands in for the input transport, which calls the filter tap for each block."""

    def __init__(self, audio_filter: _TapFilter):
        super().__init__()
        self.audio_filter = audio_filter

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and self.audio_filter.tap is not None:
            samples = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
            for block in samples.reshape(-1, 160):
                self.audio_filter.tap(block, frame.sample_rate)
        await self.push_frame(frame, direction)


def _metrics_of_type(frames, metrics_type):
    return [
        d
        for f in frames
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, metrics_type)
    ]


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICTytoProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from pipecat.processors.audio.aic_tyto_processor import AICTytoProcessor

        cls.AICTytoProcessor = AICTytoProcessor

    def setUp(self):
        self.model = MockModel(model_id="tyto-test", optimal_block_size=160)
        self.collector = MockCollector()
        self.analyzer = MockAnalyzer()

        self.enterContext(patch(f"{AIC_TYTO_MODULE}.set_sdk_id"))
        manager = self.enterContext(patch(f"{AIC_TYTO_MODULE}.AICModelManager"))
        manager.acquire = AsyncMock(return_value=(self.model, "test-cache-key"))
        self.release = manager.release
        self.enterContext(
            patch(
                f"{AIC_TYTO_MODULE}.analyzer_pair",
                side_effect=lambda model, key: (self.collector, self.analyzer),
            )
        )

    def _create(self, **kwargs):
        kwargs.setdefault("license_key", "test-key")
        processor = self.AICTytoProcessor(**kwargs)
        self.results = []
        self.summaries = []

        @processor.event_handler("on_analysis_result")
        async def on_analysis_result(processor, data):
            self.results.append(data)

        @processor.event_handler("on_call_summary")
        async def on_call_summary(processor, data):
            self.summaries.append(data)

        return processor

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            self.AICTytoProcessor(license_key="k", model_id=None, model_path=None)
        with self.assertRaises(ValueError):
            self.AICTytoProcessor(license_key="k", degraded_threshold=1.5)

    async def test_no_analysis_before_window_is_full(self):
        processor = self._create()

        await run_test(processor, frames_to_send=[_audio_frame(3.0), SleepFrame(sleep=0.3)])

        self.assertEqual(self.analyzer.analyze_calls, 0)
        self.assertEqual(self.results, [])
        self.assertEqual(self.summaries, [])

    async def test_analysis_triggers_on_audio_seconds(self):
        processor = self._create()

        # The sleeps add no audio. They only let each analysis finish.
        await run_test(
            processor,
            frames_to_send=[
                *[_audio_frame(1.0) for _ in range(5)],
                SleepFrame(sleep=0.1),
                *[_audio_frame(1.0) for _ in range(5)],
                SleepFrame(sleep=0.1),
                _audio_frame(1.0),
            ],
        )

        self.assertEqual([r.audio_secs for r in self.results], [5.0, 10.0])

    async def test_restart_resets_session(self):
        processor = self._create()

        await run_test(processor, frames_to_send=[_audio_frame(6.0), SleepFrame(sleep=0.1)])
        await run_test(processor, frames_to_send=[_audio_frame(6.0), SleepFrame(sleep=0.1)])

        self.assertEqual([s.value.windows for s in self.summaries], [1, 1])
        self.assertEqual(len(self.collector.configs), 2)

    async def test_window_result_event_and_metrics(self):
        self.analyzer.results = [MockAnalysisResult(risk_score=0.4, noise=0.6)]
        processor = self._create()

        down, _ = await run_test(
            processor,
            frames_to_send=[_audio_frame(6.0), SleepFrame(sleep=0.3)],
            pipeline_params=PipelineParams(enable_metrics=True),
        )

        self.assertGreater(len(self.results), 0)
        result = self.results[0]
        self.assertIsInstance(result, AudioQualityMetricsData)
        self.assertEqual(result.model, "tyto-test")
        self.assertEqual(result.audio_secs, 6.0)
        self.assertEqual(result.value.risk_score, 0.4)
        self.assertEqual(result.value.noise, 0.6)
        self.assertEqual(len(_metrics_of_type(down, AudioQualityMetricsData)), len(self.results))

    async def test_no_metrics_frames_when_metrics_disabled(self):
        processor = self._create()

        down, _ = await run_test(
            processor, frames_to_send=[_audio_frame(6.0), SleepFrame(sleep=0.3)]
        )

        self.assertGreater(len(self.results), 0)
        self.assertEqual(len(self.summaries), 1)
        self.assertEqual(_metrics_of_type(down, AudioQualityMetricsData), [])
        self.assertEqual(_metrics_of_type(down, AudioQualitySummaryMetricsData), [])

    async def test_summary_pushed_before_end_frame(self):
        processor = self._create()

        down, _ = await run_test(
            processor,
            frames_to_send=[_audio_frame(6.0), SleepFrame(sleep=0.3), EndFrame()],
            pipeline_params=PipelineParams(enable_metrics=True),
            send_end_frame=False,
        )

        self.assertEqual(len(self.summaries), 1)
        summary_index = next(
            i
            for i, f in enumerate(down)
            if isinstance(f, MetricsFrame)
            and any(isinstance(d, AudioQualitySummaryMetricsData) for d in f.data)
        )
        end_index = next(i for i, f in enumerate(down) if isinstance(f, EndFrame))
        self.assertLess(summary_index, end_index)
        self.assertEqual(self.summaries[0].value.windows, len(self.results))

    async def test_slow_summary_handler_completes_after_end_frame(self):
        processor = self._create()
        slow_summaries = []

        # The handler outlasts the EndFrame. Cleanup must wait for it.
        @processor.event_handler("on_call_summary")
        async def on_slow_call_summary(processor, data):
            await asyncio.sleep(0.3)
            slow_summaries.append(data)

        await run_test(processor, frames_to_send=[_audio_frame(6.0), SleepFrame(sleep=0.3)])

        self.assertEqual(len(slow_summaries), 1)

    async def test_summary_sent_once(self):
        processor = self._create()
        processor._model = self.model
        processor._windows = [
            AudioQualityScores(**dict.fromkeys(AudioQualityScores.model_fields, 0.1))
        ]

        await processor._send_summary(push_metrics=False)
        await processor._send_summary(push_metrics=False)
        # Event handlers run in tasks.
        await asyncio.sleep(0.01)

        self.assertEqual(len(self.summaries), 1)

    def test_summarize(self):
        processor = self._create(degraded_threshold=0.5)

        def scores(risk, noise, loudness):
            values = dict.fromkeys(AudioQualityScores.model_fields, 0.0)
            values.update(risk_score=risk, noise=noise, speaker_loudness=loudness)
            return AudioQualityScores(**values)

        processor._windows = [scores(0.2, 0.4, 0.9), scores(0.6, 0.2, 0.9)]

        summary = processor._summarize()

        self.assertEqual(summary.windows, 2)
        self.assertAlmostEqual(summary.mean.risk_score, 0.4)
        self.assertAlmostEqual(summary.mean.noise, 0.3)
        self.assertAlmostEqual(summary.max.risk_score, 0.6)
        self.assertAlmostEqual(summary.p95.risk_score, 0.58)
        self.assertEqual(summary.degraded_threshold, 0.5)
        self.assertEqual(summary.degraded_fraction, 0.5)
        # speaker_loudness has the highest mean but is not a degradation.
        self.assertEqual(summary.dominant_dimension, "noise")

    async def test_standalone_mode_buffers_fixed_blocks(self):
        processor = self._create()
        frames = [_audio_frame(100 / SAMPLE_RATE) for _ in range(5)]

        await run_test(processor, frames_to_send=frames)

        self.assertEqual(len(self.collector.configs), 1)
        self.assertEqual(self.collector.configs[0].sample_rate, SAMPLE_RATE)
        self.assertEqual(self.collector.configs[0].block_size, 160)
        self.assertEqual(len(self.collector.blocks), 3)
        self.assertTrue(all(len(b) == 160 for b in self.collector.blocks))

    async def test_filter_mode_attaches_tap_and_skips_buffering(self):
        audio_filter = MagicMock()
        processor = self._create(audio_filter=audio_filter)

        await run_test(processor, frames_to_send=[_audio_frame(1.0)])

        self.assertEqual(self.collector.blocks, [])
        self.assertEqual(
            [c.args for c in audio_filter.set_pre_process_tap.call_args_list],
            [(processor._on_filter_block,), (None,)],
        )

    async def test_filter_tap_feeds_collector_and_reattaches_on_start(self):
        audio_filter = _TapFilter()
        processor = self._create(audio_filter=audio_filter)
        # The sleep lets the StartFrame reach the processor before the audio.
        frames = [SleepFrame(sleep=0.05), _audio_frame(320 / SAMPLE_RATE)]

        for _ in range(2):
            await run_test(Pipeline([_TapFeeder(audio_filter), processor]), frames_to_send=frames)

        self.assertEqual(
            [(c.sample_rate, c.block_size) for c in self.collector.configs],
            [(SAMPLE_RATE, 160), (SAMPLE_RATE, 160)],
        )
        self.assertEqual(len(self.collector.blocks), 4)
        self.assertEqual(audio_filter.taps, [processor._on_filter_block, None] * 2)

    async def test_analysis_error_stops_analysis(self):
        self.analyzer.error = RuntimeError("processing not allowed")
        processor = self._create()

        _, up = await run_test(processor, frames_to_send=[_audio_frame(6.0), SleepFrame(sleep=0.3)])

        self.assertEqual(self.analyzer.analyze_calls, 1)
        self.assertTrue(any(isinstance(f, ErrorFrame) for f in up))
        self.assertEqual(self.results, [])

    async def test_no_result_after_cancel_frame(self):
        self.analyzer.gate = threading.Event()
        processor = self._create()
        probe = _CancelProbe(self.analyzer.gate)

        await run_test(
            Pipeline([processor, probe]),
            frames_to_send=[_audio_frame(6.0), SleepFrame(sleep=0.1), CancelFrame()],
            send_end_frame=False,
        )

        cancel_index = next(i for i, f in enumerate(probe.frames) if isinstance(f, CancelFrame))
        after_cancel = probe.frames[cancel_index + 1 :]
        self.assertFalse(any(isinstance(f, AudioQualityFrame) for f in after_cancel))
        self.assertEqual(self.results, [])

    async def test_cleanup_ends_session_and_releases_model(self):
        processor = self._create()

        await run_test(processor, frames_to_send=[])

        self.assertTrue(self.analyzer.terminated)
        self.release.assert_called_once_with("test-cache-key")


if __name__ == "__main__":
    unittest.main()
