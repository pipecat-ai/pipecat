#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Check if aic_sdk is available
aic_sdk: Any
try:
    import aic_sdk

    HAS_AIC_SDK = True
except ImportError:
    aic_sdk = None
    HAS_AIC_SDK = False

from tests.aic_mocks import MockModel, MockProcessorAsync, MockVadAsync  # noqa: E402

AIC_FILTER_MODULE = "pipecat.audio.filters.aic_filter"


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICFilterVADAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Test suite for AICFilterVADAnalyzer against a mocked AICFilter."""

    @classmethod
    def setUpClass(cls):
        from pipecat.audio.filters.aic_filter import AICFilter
        from pipecat.audio.vad.aic_filter_vad import AICFilterVADAnalyzer

        cls.AICFilter = AICFilter
        cls.AICFilterVADAnalyzer = AICFilterVADAnalyzer

    def setUp(self):
        self.mock_model = MockModel()
        self.mock_processor = MockProcessorAsync()
        self.mock_vad = MockVadAsync()

    def _create_filter(self, **kwargs):
        filter_kwargs = {
            "license_key": "test-key",
            "model_id": "test-model",
            "vad_model_id": "test-vad-model",
        }
        filter_kwargs.update(kwargs)
        with patch(f"{AIC_FILTER_MODULE}.set_sdk_id"):
            return self.AICFilter(**filter_kwargs)

    async def _start_filter(self, filter_instance, sample_rate=16000):
        with (
            patch(f"{AIC_FILTER_MODULE}.AICModelManager") as mock_manager_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorAsync", return_value=self.mock_processor),
            patch(f"{AIC_FILTER_MODULE}.VadAsync", return_value=self.mock_vad),
        ):
            mock_manager_cls.acquire = AsyncMock(return_value=(self.mock_model, "test-cache-key"))
            mock_config_cls.optimal.return_value = MagicMock()
            await filter_instance.start(sample_rate)

    # --- Construction --------------------------------------------------------

    def test_requires_filter_with_vad_model(self):
        """A filter without a VAD model is rejected at construction, not at poll time."""
        filter_instance = self._create_filter(vad_model_id=None)

        with self.assertRaises(ValueError) as ctx:
            self.AICFilterVADAnalyzer(aic_filter=filter_instance)

        self.assertIn("VAD model", str(ctx.exception))

    def test_accepts_filter_with_vad_model(self):
        """A VAD-configured filter constructs the analyzer."""
        analyzer = self.AICFilterVADAnalyzer(aic_filter=self._create_filter())
        self.assertIsNotNone(analyzer)

    def test_accepts_filter_with_vad_model_path(self):
        """vad_model_path satisfies the same requirement as vad_model_id."""
        from pathlib import Path

        filter_instance = self._create_filter(
            vad_model_id=None, vad_model_path=Path("/tmp/vad.aicmodel")
        )
        analyzer = self.AICFilterVADAnalyzer(aic_filter=filter_instance)
        self.assertIsNotNone(analyzer)

    # --- num_frames_required -------------------------------------------------

    def test_num_frames_required_before_filter_start(self):
        """Before the filter starts, fall back to a 10 ms window."""
        analyzer = self.AICFilterVADAnalyzer(aic_filter=self._create_filter())
        self.assertEqual(analyzer.num_frames_required(), 160)

    def test_num_frames_required_before_start_with_sample_rate(self):
        """The fallback scales with the sample rate once the pipeline sets it."""
        analyzer = self.AICFilterVADAnalyzer(aic_filter=self._create_filter())
        analyzer.set_sample_rate(24000)
        self.assertEqual(analyzer.num_frames_required(), 240)

    async def test_num_frames_required_tracks_filter_block(self):
        """Once the filter starts, polling matches its block size."""
        self.mock_model = MockModel(optimal_block_size=480)
        filter_instance = self._create_filter()
        analyzer = self.AICFilterVADAnalyzer(aic_filter=filter_instance)

        await self._start_filter(filter_instance)

        self.assertEqual(filter_instance.frames_per_block, 480)
        self.assertEqual(analyzer.num_frames_required(), 480)

    # --- voice_confidence ----------------------------------------------------

    async def test_voice_confidence_before_filter_start(self):
        """Polling a not-yet-started filter reports no speech instead of raising."""
        analyzer = self.AICFilterVADAnalyzer(aic_filter=self._create_filter())
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)

    async def test_voice_confidence_reads_filter_vad(self):
        """The analyzer surfaces the filter's VAD probability."""
        filter_instance = self._create_filter()
        analyzer = self.AICFilterVADAnalyzer(aic_filter=filter_instance)
        await self._start_filter(filter_instance)

        self.mock_vad.vad_ctx.raw_probability = 0.73
        self.assertAlmostEqual(analyzer.voice_confidence(b"\x00" * 320), 0.73)

    async def test_voice_confidence_ignores_its_own_buffer(self):
        """The analyzer's input buffer is not fed to any VAD; the filter already did that."""
        filter_instance = self._create_filter()
        analyzer = self.AICFilterVADAnalyzer(aic_filter=filter_instance)
        await self._start_filter(filter_instance)

        self.mock_vad.vad_ctx.raw_probability = 0.5
        analyzer.voice_confidence(b"\xff" * 320)

        self.assertEqual(self.mock_vad.process_calls, [])

    async def test_voice_confidence_clamps_out_of_range(self):
        """Probabilities outside [0.0, 1.0] are clamped to the VADAnalyzer range."""
        filter_instance = self._create_filter()
        analyzer = self.AICFilterVADAnalyzer(aic_filter=filter_instance)
        await self._start_filter(filter_instance)

        self.mock_vad.vad_ctx.raw_probability = 1.4
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 1.0)
        self.mock_vad.vad_ctx.raw_probability = -0.3
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)

    async def test_voice_confidence_swallows_sdk_errors(self):
        """An SDK read error returns 0.0 so the pipeline stays alive."""
        filter_instance = self._create_filter()
        analyzer = self.AICFilterVADAnalyzer(aic_filter=filter_instance)
        await self._start_filter(filter_instance)

        self.mock_vad.vad_ctx.raise_on_detect = True
        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)

    async def test_voice_confidence_after_filter_stop(self):
        """Polling after the filter stops reports no speech instead of raising."""
        filter_instance = self._create_filter()
        analyzer = self.AICFilterVADAnalyzer(aic_filter=filter_instance)
        await self._start_filter(filter_instance)
        await filter_instance.stop()

        self.assertEqual(analyzer.voice_confidence(b"\x00" * 320), 0.0)


if __name__ == "__main__":
    unittest.main()
