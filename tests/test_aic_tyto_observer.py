#
# Copyright (c) 2026, Daily
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

from pipecat.frames.frames import CancelFrame, EndFrame, InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.aic_mocks import MockAnalysisResult, MockAnalyzer, MockCollector, MockModel

AIC_TYTO_MODULE = "pipecat.observers.aic_tyto_observer"

SAMPLE_RATE = 16000


def _audio_frame(secs: float) -> InputAudioRawFrame:
    samples = np.zeros(int(secs * SAMPLE_RATE), dtype=np.int16)
    return InputAudioRawFrame(audio=samples.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1)


class _CancelProbe(FrameProcessor):
    """Releases ``gate`` while a CancelFrame push is in progress."""

    def __init__(self, gate: threading.Event):
        super().__init__(enable_direct_mode=True)
        self.gate = gate

    async def process_frame(self, frame, direction: FrameDirection):
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


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICTytoObserver(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from pipecat.observers.aic_tyto_observer import AICAudioQualityScores, AICTytoObserver

        cls.AICAudioQualityScores = AICAudioQualityScores
        cls.AICTytoObserver = AICTytoObserver

    def setUp(self):
        self.model = MockModel(model_id="tyto-test", optimal_block_size=160)
        self.collector = MockCollector()
        self.analyzer = MockAnalyzer()

        self.enterContext(patch(f"{AIC_TYTO_MODULE}.set_sdk_id"))
        self.manager = self.enterContext(patch(f"{AIC_TYTO_MODULE}.AICModelManager"))
        self.manager.acquire = AsyncMock(return_value=(self.model, "test-cache-key"))
        self.enterContext(
            patch(
                f"{AIC_TYTO_MODULE}.analyzer_pair",
                side_effect=lambda model, key: (self.collector, self.analyzer),
            )
        )

    def _create(self, **kwargs):
        kwargs.setdefault("license_key", "test-key")
        observer = self.AICTytoObserver(**kwargs)
        self.results = []
        self.summaries = []

        @observer.event_handler("on_analysis_result")
        async def on_analysis_result(observer, scores, audio_secs):
            self.results.append((scores, audio_secs))

        @observer.event_handler("on_call_summary")
        async def on_call_summary(observer, summary):
            self.summaries.append(summary)

        return observer

    async def _run(self, observer, frames, processor=None, **kwargs):
        return await run_test(
            processor or IdentityFilter(), frames_to_send=frames, observers=[observer], **kwargs
        )

    def _scores(self, **values):
        return self.AICAudioQualityScores(
            **{**dict.fromkeys(self.AICAudioQualityScores.model_fields, 0.0), **values}
        )

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            self.AICTytoObserver(license_key="k", model_id=None, model_path=None)
        with self.assertRaises(ValueError):
            self.AICTytoObserver(license_key="k", degraded_threshold=1.5)

    async def test_no_analysis_before_window_is_full(self):
        observer = self._create()

        await self._run(observer, [_audio_frame(3.0), SleepFrame(sleep=0.3)])

        self.assertEqual(self.analyzer.analyze_calls, 0)
        self.assertEqual(self.results, [])
        self.assertEqual(self.summaries, [])

    async def test_analysis_triggers_on_audio_seconds(self):
        observer = self._create()

        # The sleeps add no audio. They only let each analysis finish.
        await self._run(
            observer,
            [
                *[_audio_frame(1.0) for _ in range(5)],
                SleepFrame(sleep=0.1),
                *[_audio_frame(1.0) for _ in range(5)],
                SleepFrame(sleep=0.1),
                _audio_frame(1.0),
            ],
        )

        self.assertEqual([audio_secs for _, audio_secs in self.results], [5.0, 10.0])

    async def test_each_run_starts_a_new_session(self):
        observer = self._create()

        await self._run(observer, [_audio_frame(6.0), SleepFrame(sleep=0.1)])
        await self._run(observer, [_audio_frame(6.0), SleepFrame(sleep=0.1)])

        self.assertEqual([s.windows for s in self.summaries], [1, 1])
        self.assertEqual(len(self.collector.configs), 2)

    async def test_window_result_event(self):
        self.analyzer.results = [MockAnalysisResult(risk_score=0.4, noise=0.6)]
        observer = self._create()

        await self._run(observer, [_audio_frame(6.0), SleepFrame(sleep=0.3)])

        scores, audio_secs = self.results[0]
        self.assertEqual(audio_secs, 6.0)
        self.assertEqual(scores.risk_score, 0.4)
        self.assertEqual(scores.noise, 0.6)

    async def test_summary_on_end_frame(self):
        observer = self._create()

        await self._run(
            observer,
            [_audio_frame(6.0), SleepFrame(sleep=0.3), EndFrame()],
            send_end_frame=False,
        )

        self.assertEqual(len(self.summaries), 1)
        self.assertEqual(self.summaries[0].windows, len(self.results))

    async def test_summary_on_cleanup(self):
        observer = self._create()
        await observer.setup(TaskManager())
        observer._windows = [self._scores(risk_score=0.1)]

        await observer.cleanup()

        self.assertEqual(len(self.summaries), 1)

    async def test_slow_summary_handler_completes_before_cleanup_returns(self):
        observer = self._create()
        slow_summaries = []

        @observer.event_handler("on_call_summary")
        async def on_slow_call_summary(observer, summary):
            await asyncio.sleep(0.3)
            slow_summaries.append(summary)

        await self._run(observer, [_audio_frame(6.0), SleepFrame(sleep=0.3)])

        self.assertEqual(len(slow_summaries), 1)

    async def test_summary_sent_once(self):
        observer = self._create()
        observer._windows = [self._scores(risk_score=0.1)]

        await observer._finish()
        await observer._finish()
        # Event handlers run in tasks.
        await asyncio.sleep(0.01)

        self.assertEqual(len(self.summaries), 1)

    def test_summarize(self):
        observer = self._create(degraded_threshold=0.5)
        observer._windows = [
            self._scores(risk_score=0.2, noise=0.4, speaker_loudness=0.9),
            self._scores(risk_score=0.6, noise=0.2, speaker_loudness=0.9),
        ]

        summary = observer._summarize()

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
        observer = self._create()

        await self._run(observer, [_audio_frame(100 / SAMPLE_RATE) for _ in range(5)])

        self.assertEqual(len(self.collector.configs), 1)
        self.assertEqual(self.collector.configs[0].sample_rate, SAMPLE_RATE)
        self.assertEqual(self.collector.configs[0].block_size, 160)
        self.assertEqual(len(self.collector.blocks), 3)
        self.assertTrue(all(len(b) == 160 for b in self.collector.blocks))

    async def test_filter_mode_attaches_tap_and_skips_buffering(self):
        audio_filter = MagicMock()
        observer = self._create(audio_filter=audio_filter)

        await self._run(observer, [_audio_frame(1.0)])

        self.assertEqual(self.collector.blocks, [])
        self.assertEqual(
            [c.args for c in audio_filter.set_pre_process_tap.call_args_list],
            [(observer._on_filter_block,), (None,)],
        )

    async def test_filter_tap_feeds_collector_and_reattaches_on_setup(self):
        audio_filter = _TapFilter()
        observer = self._create(audio_filter=audio_filter)

        for _ in range(2):
            await self._run(
                observer, [_audio_frame(320 / SAMPLE_RATE)], processor=_TapFeeder(audio_filter)
            )

        self.assertEqual(
            [(c.sample_rate, c.block_size) for c in self.collector.configs],
            [(SAMPLE_RATE, 160), (SAMPLE_RATE, 160)],
        )
        self.assertEqual(len(self.collector.blocks), 4)
        self.assertEqual(audio_filter.taps, [observer._on_filter_block, None] * 2)

    async def test_filter_mode_analyzes_only_tapped_audio(self):
        audio_filter = _TapFilter()
        observer = self._create(audio_filter=audio_filter)
        frames = [_audio_frame(6.0), SleepFrame(sleep=0.1)]

        # No feeder simulates a filter that failed to initialize.
        await self._run(observer, frames)

        self.assertEqual(self.analyzer.analyze_calls, 0)
        self.assertEqual(self.summaries, [])

        await self._run(observer, frames, processor=_TapFeeder(audio_filter))

        self.assertEqual(self.analyzer.analyze_calls, 1)
        self.assertEqual(len(self.summaries), 1)

    async def test_analysis_error_stops_analysis(self):
        self.analyzer.error = RuntimeError("processing not allowed")
        observer = self._create()

        await self._run(
            observer,
            [_audio_frame(6.0), SleepFrame(sleep=0.1), _audio_frame(6.0), SleepFrame(sleep=0.1)],
        )

        self.assertEqual(self.analyzer.analyze_calls, 1)
        self.assertEqual(self.results, [])

    async def test_no_result_after_cancel_frame(self):
        self.analyzer.gate = threading.Event()
        observer = self._create()

        await self._run(
            observer,
            [_audio_frame(6.0), SleepFrame(sleep=0.1), CancelFrame()],
            processor=Pipeline([_CancelProbe(self.analyzer.gate)]),
            send_end_frame=False,
        )

        self.assertEqual(self.analyzer.analyze_calls, 1)
        self.assertEqual(self.results, [])

    async def test_setup_failure_skips_analysis(self):
        self.manager.acquire = AsyncMock(side_effect=RuntimeError("download failed"))
        observer = self._create()

        await self._run(observer, [_audio_frame(6.0), SleepFrame(sleep=0.1)])

        self.assertEqual(self.analyzer.analyze_calls, 0)
        self.assertEqual(self.summaries, [])
        self.manager.release.assert_not_called()

    async def test_cleanup_ends_session_and_releases_model(self):
        observer = self._create()

        await self._run(observer, [])

        self.assertTrue(self.analyzer.terminated)
        self.manager.release.assert_called_once_with("test-cache-key")


if __name__ == "__main__":
    unittest.main()
