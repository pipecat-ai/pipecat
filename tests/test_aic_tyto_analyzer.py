#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

# Check if aic_sdk is available
aic_sdk: Any
try:
    import aic_sdk

    HAS_AIC_SDK = True
except ImportError:
    aic_sdk = None
    HAS_AIC_SDK = False

from tests.aic_mocks import (  # noqa: E402
    MockAnalysisResult,
    MockAnalyzer,
    MockCollector,
    MockModel,
)

# Module path for patching
TYTO_MODULE = "pipecat.processors.audio.aic_tyto_analyzer"


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICTytoAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Test suite for AICTytoAnalyzer using mocked aic_sdk types."""

    @classmethod
    def setUpClass(cls):
        from pipecat.metrics.metrics import AICAudioQualityMetricsData
        from pipecat.processors.audio.aic_tyto_analyzer import (
            DEFAULT_TYTO_MODEL_ID,
            AICTytoAnalyzer,
        )

        cls.AICTytoAnalyzer = AICTytoAnalyzer
        cls.DEFAULT_TYTO_MODEL_ID = DEFAULT_TYTO_MODEL_ID
        cls.AICAudioQualityMetricsData = AICAudioQualityMetricsData

    def setUp(self):
        self.mock_model = MockModel(model_id="tyto-l-16khz")
        self.mock_collector = MockCollector()
        self.mock_analyzer = MockAnalyzer()

        # Patch every aic_sdk touchpoint for the whole test, so lazy collector
        # init inside _buffer_audio is mocked too (not just __init__).
        patchers = {
            "set_sdk_id": patch(f"{TYTO_MODULE}.set_sdk_id"),
            "Model": patch(f"{TYTO_MODULE}.Model"),
            "ProcessorConfig": patch(f"{TYTO_MODULE}.ProcessorConfig"),
            "analyzer_pair": patch(
                f"{TYTO_MODULE}.analyzer_pair",
                return_value=(self.mock_collector, self.mock_analyzer),
            ),
        }
        self.mocks = {name: p.start() for name, p in patchers.items()}
        for p in patchers.values():
            self.addCleanup(p.stop)

        self.mocks["Model"].from_file.return_value = self.mock_model
        self.mocks["Model"].download.return_value = "/tmp/test.aicmodel"
        self.mocks["ProcessorConfig"].optimal.return_value = MagicMock(name="config")

    def _make(self, **kwargs):
        analyzer_kwargs = {"license_key": "test-key"}
        analyzer_kwargs.update(kwargs)
        return self.AICTytoAnalyzer(**analyzer_kwargs)

    @staticmethod
    def _audio_frame(*, num_samples: int, sample_rate: int = 16000, num_channels: int = 1):
        from pipecat.frames.frames import InputAudioRawFrame

        audio = (np.arange(num_samples, dtype=np.int16)).tobytes()
        return InputAudioRawFrame(audio=audio, sample_rate=sample_rate, num_channels=num_channels)

    # --- Construction --------------------------------------------------------

    def test_requires_model_id_or_path(self):
        """Neither model_id nor model_path → ValueError."""
        with self.assertRaises(ValueError):
            self._make(model_id=None, model_path=None)

    def test_default_model_id(self):
        """Default model_id is the published Tyto model."""
        analyzer = self._make()
        self.assertEqual(analyzer._model_id, "tyto-l-16khz")
        self.assertEqual(analyzer._model_id, self.DEFAULT_TYTO_MODEL_ID)

    def test_default_download_dir(self):
        """Default model_download_dir lives under the user's cache."""
        analyzer = self._make()
        expected = Path.home() / ".cache" / "pipecat" / "aic-models"
        self.assertEqual(analyzer._model_download_dir, expected)

    def test_eager_loads_model_via_model_id(self):
        """__init__ registers telemetry and downloads/loads the model."""
        self._make()
        self.mocks["set_sdk_id"].assert_called_once_with(6)
        self.mocks["Model"].download.assert_called_once_with(
            "tyto-l-16khz",
            str(Path.home() / ".cache" / "pipecat" / "aic-models"),
        )
        self.mocks["Model"].from_file.assert_called_once_with("/tmp/test.aicmodel")

    def test_eager_loads_model_via_model_path(self):
        """model_path skips the download step entirely."""
        self._make(model_id=None, model_path=Path("/tmp/custom.aicmodel"))
        self.mocks["Model"].download.assert_not_called()
        self.mocks["Model"].from_file.assert_called_once_with("/tmp/custom.aicmodel")

    def test_init_shuts_down_executor_on_eager_load_failure(self):
        """A failed eager load tears down the executor so no worker thread leaks."""
        mock_executor = MagicMock()
        with (
            patch(f"{TYTO_MODULE}.ThreadPoolExecutor", return_value=mock_executor),
            patch.object(self.mocks["Model"], "download", side_effect=RuntimeError("network")),
        ):
            with self.assertRaises(RuntimeError):
                self._make()
        mock_executor.shutdown.assert_called_once_with(wait=False)

    # --- Audio buffering -----------------------------------------------------

    def test_buffer_initializes_collector_with_variable_frames(self):
        """First audio frame lazily builds the collector with allow_variable_frames."""
        analyzer = self._make()
        analyzer._buffer_audio(self._audio_frame(num_samples=160))

        self.mocks["analyzer_pair"].assert_called_once()
        self.mocks["ProcessorConfig"].optimal.assert_called_once()
        _, kwargs = self.mocks["ProcessorConfig"].optimal.call_args
        self.assertTrue(kwargs["allow_variable_frames"])
        self.assertEqual(kwargs["sample_rate"], 16000)
        self.assertEqual(kwargs["num_channels"], 1)
        self.assertEqual(len(self.mock_collector.buffer_calls), 1)
        self.assertEqual(self.mock_collector.buffer_calls[0].shape, (1, 160))
        self.assertEqual(self.mock_collector.buffer_calls[0].dtype, np.float32)

    def test_buffer_normalizes_int16(self):
        """int16 samples are scaled into [-1.0, 1.0) float32."""
        analyzer = self._make()
        analyzer._buffer_audio(self._audio_frame(num_samples=4))
        buffered = self.mock_collector.buffer_calls[0]
        # Samples were 0,1,2,3 → divided by 32768.
        np.testing.assert_allclose(buffered[0], np.array([0, 1, 2, 3], dtype=np.float32) / 32768.0)

    def test_buffer_deinterleaves_multichannel(self):
        """Stereo audio is reshaped to (channels, frames)."""
        analyzer = self._make()
        # 8 interleaved samples → 2 channels × 4 frames.
        analyzer._buffer_audio(self._audio_frame(num_samples=8, num_channels=2))
        self.assertEqual(self.mock_collector.buffer_calls[0].shape, (2, 4))

    def test_buffer_reinitializes_on_sample_rate_change(self):
        """A changed input sample rate rebuilds the collector."""
        analyzer = self._make()
        analyzer._buffer_audio(self._audio_frame(num_samples=160, sample_rate=16000))
        analyzer._buffer_audio(self._audio_frame(num_samples=80, sample_rate=8000))
        self.assertEqual(self.mocks["analyzer_pair"].call_count, 2)

    def test_buffer_does_not_reinitialize_on_same_config(self):
        """Repeated frames at the same rate reuse the collector."""
        analyzer = self._make()
        analyzer._buffer_audio(self._audio_frame(num_samples=160))
        analyzer._buffer_audio(self._audio_frame(num_samples=160))
        self.assertEqual(self.mocks["analyzer_pair"].call_count, 1)
        self.assertEqual(len(self.mock_collector.buffer_calls), 2)

    def test_buffer_swallows_sdk_errors(self):
        """A buffering error is latched and swallowed (pipeline stays alive)."""
        analyzer = self._make()
        self.mock_collector.raise_on_buffer = True
        analyzer._buffer_audio(self._audio_frame(num_samples=160))  # must not raise
        self.assertTrue(analyzer._analysis_error_logged)

    # --- Analysis ------------------------------------------------------------

    def test_build_metrics_maps_all_scores(self):
        """_build_metrics copies the 7 Tyto scores plus processor/model."""
        analyzer = self._make()
        result = MockAnalysisResult(
            risk_score=0.9,
            speaker_reverb=0.1,
            speaker_loudness=0.5,
            interfering_speech=0.2,
            media_speech=0.3,
            noise=0.4,
            packet_loss=0.05,
        )
        data = analyzer._build_metrics(result)
        self.assertIsInstance(data, self.AICAudioQualityMetricsData)
        self.assertEqual(data.processor, analyzer.name)
        self.assertEqual(data.model, "tyto-l-16khz")
        self.assertAlmostEqual(data.risk_score, 0.9)
        self.assertAlmostEqual(data.speaker_reverb, 0.1)
        self.assertAlmostEqual(data.speaker_loudness, 0.5)
        self.assertAlmostEqual(data.interfering_speech, 0.2)
        self.assertAlmostEqual(data.media_speech, 0.3)
        self.assertAlmostEqual(data.noise, 0.4)
        self.assertAlmostEqual(data.packet_loss, 0.05)

    async def test_analyze_once_emits_metrics_frame(self):
        """A successful analysis pushes a MetricsFrame with the scores."""
        from pipecat.frames.frames import MetricsFrame

        analyzer = self._make()
        analyzer._analyzer = MockAnalyzer(result=MockAnalysisResult(risk_score=0.8))
        analyzer.push_frame = AsyncMock()
        await analyzer._analyze_once()
        analyzer.push_frame.assert_awaited_once()
        pushed = analyzer.push_frame.await_args.args[0]
        self.assertIsInstance(pushed, MetricsFrame)
        self.assertIsInstance(pushed.data[0], self.AICAudioQualityMetricsData)
        self.assertAlmostEqual(pushed.data[0].risk_score, 0.8)

    async def test_analyze_once_fires_event(self):
        """on_audio_analysis handlers receive the metrics data."""
        analyzer = self._make()
        analyzer._analyzer = MockAnalyzer(result=MockAnalysisResult(noise=0.6))
        analyzer.push_frame = AsyncMock()
        captured = []

        @analyzer.event_handler("on_audio_analysis")
        async def _on(_proc, data):
            captured.append(data)

        await analyzer._analyze_once()
        # The event handler runs as a fire-and-forget task; let it run.
        import asyncio

        await asyncio.sleep(0)
        self.assertEqual(len(captured), 1)
        self.assertAlmostEqual(captured[0].noise, 0.6)

    async def test_analyze_once_without_analyzer_is_noop(self):
        """No analysis runs before the collector/analyzer is initialized."""
        analyzer = self._make()
        analyzer._analyzer = None
        analyzer.push_frame = AsyncMock()
        await analyzer._analyze_once()
        analyzer.push_frame.assert_not_called()

    async def test_analyze_once_swallows_sdk_errors(self):
        """An analysis error is latched and swallowed; nothing is pushed."""
        analyzer = self._make()
        analyzer._analyzer = MockAnalyzer(raise_on_analyze=True)
        analyzer.push_frame = AsyncMock()
        await analyzer._analyze_once()
        analyzer.push_frame.assert_not_called()
        self.assertTrue(analyzer._analysis_error_logged)

    # --- Lifecycle -----------------------------------------------------------

    def test_start_spawns_task_once(self):
        """_start spawns the analysis loop exactly once."""
        analyzer = self._make()

        def _fake_create_task(coro, *args, **kwargs):
            coro.close()  # avoid an un-awaited-coroutine warning
            return MagicMock()

        analyzer.create_task = MagicMock(side_effect=_fake_create_task)
        analyzer._start()
        analyzer._start()
        analyzer.create_task.assert_called_once()

    async def test_cleanup_releases_handles(self):
        """cleanup() shuts the executor and nils SDK handles."""
        analyzer = self._make()
        analyzer._collector = self.mock_collector
        analyzer._analyzer = self.mock_analyzer
        await analyzer.cleanup()
        self.assertIsNone(analyzer._collector)
        self.assertIsNone(analyzer._analyzer)
        self.assertIsNone(analyzer._model)

    async def test_cleanup_cancels_analysis_task(self):
        """cleanup() cancels a running analysis task."""
        analyzer = self._make()
        analyzer.cancel_task = AsyncMock()
        sentinel = MagicMock()
        analyzer._analysis_task = sentinel
        await analyzer.cleanup()
        analyzer.cancel_task.assert_awaited_once_with(sentinel)
        self.assertIsNone(analyzer._analysis_task)

    # --- Integration (passive tap) ------------------------------------------

    async def test_forwards_frames_and_taps_audio(self):
        """Frames pass through unchanged while input audio is buffered."""
        from pipecat.frames.frames import InputAudioRawFrame, TextFrame
        from pipecat.tests.utils import run_test

        # Large interval so the analysis loop never fires during the test.
        analyzer = self._make(analysis_interval=100.0)
        audio = self._audio_frame(num_samples=160)
        received_down, _ = await run_test(
            analyzer,
            frames_to_send=[audio, TextFrame("hello")],
            expected_down_frames=[InputAudioRawFrame, TextFrame],
        )
        # The input audio frame was tapped into the collector.
        self.assertGreaterEqual(len(self.mock_collector.buffer_calls), 1)


if __name__ == "__main__":
    unittest.main()
