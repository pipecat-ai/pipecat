#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Mock hecttor_sdk before any pipecat imports so these tests run without the
# SDK installed. The package is not distributed on PyPI.
mock_hecttor_sdk = MagicMock()
sys.modules["hecttor_sdk"] = mock_hecttor_sdk

from pipecat.audio.filters.hecttor_filter import SUPPORTED_MODELS, HecttorFilter
from pipecat.frames.frames import FilterEnableFrame

API_KEY = "test-api-key"
SAMPLE_RATE = 16000
SAMPLES_PER_CHUNK = 320  # 20ms at 16kHz
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * 2


class TestHecttorFilter(unittest.IsolatedAsyncioTestCase):
    """Test suite for the HecttorFilter audio filter."""

    def setUp(self):
        """Set up a mock enhancer that echoes back the audio it is given."""
        self.mock_enhancer = MagicMock()
        self.mock_enhancer.initialize.return_value = (True, "ok")
        self.mock_enhancer.get_chunk_size_samples.return_value = SAMPLES_PER_CHUNK
        self.mock_enhancer.process_chunk.side_effect = lambda chunk: (chunk, "ok")

        self.enhancer_patcher = patch(
            "pipecat.audio.filters.hecttor_filter.ASRSpeechEnhancer",
            return_value=self.mock_enhancer,
        )
        self.enhancer_patcher.start()
        self.addCleanup(self.enhancer_patcher.stop)

        for name in ("ASRSpeechEnhancerConfig", "ModelConfig"):
            patcher = patch(f"pipecat.audio.filters.hecttor_filter.{name}")
            patcher.start()
            self.addCleanup(patcher.stop)

    async def _started_filter(self, **kwargs) -> HecttorFilter:
        """Build a filter with the API key set and start it."""
        kwargs.setdefault("api_key", API_KEY)
        audio_filter = HecttorFilter(**kwargs)
        await audio_filter.start(SAMPLE_RATE)
        return audio_filter

    @staticmethod
    def _audio(num_samples: int, value: int = 1000) -> bytes:
        """Build int16 PCM audio of a constant value."""
        return np.full(num_samples, value, dtype=np.int16).tobytes()

    # Initialization

    def test_api_key_from_argument(self):
        audio_filter = HecttorFilter(api_key=API_KEY)
        self.assertEqual(audio_filter._api_key, API_KEY)

    def test_api_key_from_environment(self):
        with patch.dict("os.environ", {"HECTTOR_API_KEY": "env-key"}):
            audio_filter = HecttorFilter()
        self.assertEqual(audio_filter._api_key, "env-key")

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                HecttorFilter()

    def test_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            HecttorFilter(api_key=API_KEY, model_name="not-a-model")

    def test_supported_models_accepted(self):
        for model_name in SUPPORTED_MODELS:
            # Models restricted to 20ms are covered by the chunk size tests.
            HecttorFilter(api_key=API_KEY, model_name=model_name, chunk_size_ms=20)

    def test_invalid_chunk_size_raises(self):
        with self.assertRaises(ValueError):
            HecttorFilter(api_key=API_KEY, chunk_size_ms=10)

    def test_model_requiring_20ms_rejects_16ms(self):
        with self.assertRaises(ValueError):
            HecttorFilter(api_key=API_KEY, model_name="coda-vi-1.0", chunk_size_ms=16)

    def test_model_allowing_16ms_is_accepted(self):
        audio_filter = HecttorFilter(api_key=API_KEY, model_name="crest-1.0", chunk_size_ms=16)
        self.assertEqual(audio_filter._chunk_size_ms, 16)

    def test_enhancer_weight_out_of_range_raises(self):
        for weight in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                HecttorFilter(api_key=API_KEY, enhancer_weight=weight)

    def test_enhancer_weight_bounds_accepted(self):
        for weight in (0.0, 0.5, 1.0):
            HecttorFilter(api_key=API_KEY, enhancer_weight=weight)

    # Lifecycle

    async def test_start_initializes_enhancer(self):
        audio_filter = await self._started_filter()
        self.mock_enhancer.initialize.assert_called_once()
        self.assertEqual(audio_filter._samples_per_chunk, SAMPLES_PER_CHUNK)

    async def test_start_failure_raises_and_clears_enhancer(self):
        self.mock_enhancer.initialize.return_value = (False, "bad key")
        audio_filter = HecttorFilter(api_key=API_KEY)
        with self.assertRaises(RuntimeError):
            await audio_filter.start(SAMPLE_RATE)
        self.assertIsNone(audio_filter._enhancer)

    async def test_stop_resets_enhancer(self):
        audio_filter = await self._started_filter()
        await audio_filter.stop()
        self.mock_enhancer.reset_caches.assert_called_once()
        self.assertIsNone(audio_filter._enhancer)

    async def test_stop_without_start(self):
        audio_filter = HecttorFilter(api_key=API_KEY)
        await audio_filter.stop()
        self.assertIsNone(audio_filter._enhancer)

    async def test_stop_clears_buffer(self):
        audio_filter = await self._started_filter()
        await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK // 2))
        self.assertGreater(len(audio_filter._audio_buffer), 0)
        await audio_filter.stop()
        self.assertEqual(len(audio_filter._audio_buffer), 0)

    # Control frames

    async def test_process_frame_toggles_filtering(self):
        audio_filter = await self._started_filter()

        await audio_filter.process_frame(FilterEnableFrame(False))
        self.assertFalse(audio_filter._filtering)

        await audio_filter.process_frame(FilterEnableFrame(True))
        self.assertTrue(audio_filter._filtering)

    # Filtering

    async def test_filter_passthrough_when_disabled(self):
        audio_filter = await self._started_filter()
        await audio_filter.process_frame(FilterEnableFrame(False))

        audio = self._audio(SAMPLES_PER_CHUNK)
        self.assertEqual(await audio_filter.filter(audio), audio)
        self.mock_enhancer.process_chunk.assert_not_called()

    async def test_filter_passthrough_before_start(self):
        audio_filter = HecttorFilter(api_key=API_KEY)
        audio = self._audio(SAMPLES_PER_CHUNK)
        self.assertEqual(await audio_filter.filter(audio), audio)

    async def test_filter_buffers_partial_chunk(self):
        audio_filter = await self._started_filter()
        result = await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK // 2))
        self.assertEqual(result, b"")
        self.mock_enhancer.process_chunk.assert_not_called()

    async def test_filter_emits_once_chunk_is_complete(self):
        audio_filter = await self._started_filter()

        self.assertEqual(await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK // 2)), b"")
        result = await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK // 2))

        self.assertEqual(len(result), BYTES_PER_CHUNK)
        self.mock_enhancer.process_chunk.assert_called_once()

    async def test_filter_processes_multiple_chunks(self):
        audio_filter = await self._started_filter()
        result = await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK * 3))

        self.assertEqual(len(result), BYTES_PER_CHUNK * 3)
        self.assertEqual(self.mock_enhancer.process_chunk.call_count, 3)

    async def test_filter_retains_remainder_in_buffer(self):
        audio_filter = await self._started_filter()
        remainder = 100
        result = await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK + remainder))

        self.assertEqual(len(result), BYTES_PER_CHUNK)
        self.assertEqual(len(audio_filter._audio_buffer), remainder * 2)

    async def test_filter_converts_to_float32_for_the_sdk(self):
        audio_filter = await self._started_filter()
        await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK, value=16384))

        chunk = self.mock_enhancer.process_chunk.call_args[0][0]
        self.assertEqual(chunk.dtype, np.float32)
        self.assertEqual(len(chunk), SAMPLES_PER_CHUNK)
        np.testing.assert_allclose(chunk, 0.5, atol=1e-4)

    async def test_filter_roundtrips_audio(self):
        audio_filter = await self._started_filter()
        value = 1000
        result = await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK, value=value))

        samples = np.frombuffer(result, dtype=np.int16)
        # int16 -> float32 -> int16 conversion is lossy by up to one LSB.
        np.testing.assert_allclose(samples, value, atol=2)

    async def test_filter_skips_warmup_chunks(self):
        audio_filter = await self._started_filter()
        self.mock_enhancer.process_chunk.side_effect = lambda chunk: (None, "warming up")

        result = await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK))
        self.assertEqual(result, b"")

    async def test_filter_clips_out_of_range_output(self):
        audio_filter = await self._started_filter()
        self.mock_enhancer.process_chunk.side_effect = lambda chunk: (
            np.full(len(chunk), 2.0, dtype=np.float32),
            "ok",
        )

        result = await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK))
        samples = np.frombuffer(result, dtype=np.int16)
        np.testing.assert_array_equal(samples, 32767)

    async def test_filter_returns_original_audio_on_error(self):
        audio_filter = await self._started_filter()
        self.mock_enhancer.process_chunk.side_effect = RuntimeError("boom")

        audio = self._audio(SAMPLES_PER_CHUNK)
        self.assertEqual(await audio_filter.filter(audio), audio)

    async def test_filter_across_start_stop_cycles(self):
        audio_filter = await self._started_filter()
        await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK // 2))
        await audio_filter.stop()

        await audio_filter.start(SAMPLE_RATE)
        # The buffer was cleared, so a half chunk is still not enough to emit.
        self.assertEqual(await audio_filter.filter(self._audio(SAMPLES_PER_CHUNK // 2)), b"")


if __name__ == "__main__":
    unittest.main()
