#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

# Check if aic_sdk is available
try:
    import aic_sdk

    HAS_AIC_SDK = True
except ImportError:
    HAS_AIC_SDK = False

# Module path for patching
AIC_FILTER_MODULE = "pipecat.audio.filters.aic_filter"


class MockProcessor:
    """A lightweight mock for AIC ProcessorAsync that mimics real behavior."""

    def __init__(self):
        self.processor_ctx = MockProcessorContext()
        self.vad_ctx = MockVadContext()

    def get_processor_context(self):
        return self.processor_ctx

    def get_vad_context(self):
        return self.vad_ctx

    async def process_async(self, audio_array):
        # Return a copy of the input (simulating passthrough)
        return audio_array.copy()


class MockProcessorContext:
    """A lightweight mock for AIC ProcessorContext."""

    def __init__(self):
        self.parameters_set: list[tuple] = []
        self.reset_called = False
        self._output_delay = 0

    def get_output_delay(self):
        return self._output_delay

    def set_parameter(self, param, value):
        self.parameters_set.append((param, value))

    def reset(self):
        self.reset_called = True


class MockVadContext:
    """A lightweight mock for AIC VadContext."""

    def __init__(self, speech_detected: bool = False):
        self.speech_detected = speech_detected
        self.parameters_set: list[tuple] = []

    def is_speech_detected(self) -> bool:
        return self.speech_detected

    def set_parameter(self, param, value):
        self.parameters_set.append((param, value))


class MockModel:
    """A lightweight mock for AIC Model."""

    def __init__(self, model_id: str = "test-model"):
        self._model_id = model_id
        self._optimal_num_frames = 160
        self._optimal_sample_rate = 16000

    def get_optimal_num_frames(self, sample_rate: int):
        """Return optimal number of frames for the given sample rate."""
        return self._optimal_num_frames

    def get_id(self):
        return self._model_id

    def get_optimal_sample_rate(self):
        return self._optimal_sample_rate


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICFilter(unittest.IsolatedAsyncioTestCase):
    """Test suite for AICFilter audio filter using real aic_sdk types."""

    @classmethod
    def setUpClass(cls):
        """Import AICFilter after confirming aic_sdk is available."""
        from pipecat.audio.filters.aic_filter import AICFilter
        from pipecat.frames.frames import FilterEnableFrame

        cls.AICFilter = AICFilter
        cls.FilterEnableFrame = FilterEnableFrame

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_model = MockModel()
        self.mock_processor = MockProcessor()

    def _create_filter_with_mocks(self, **kwargs):
        """Create an AICFilter with mocked SDK components."""
        filter_kwargs = {
            "license_key": "test-key",
            "model_id": "test-model",
        }
        filter_kwargs.update(kwargs)
        with patch(f"{AIC_FILTER_MODULE}.set_sdk_id"):
            return self.AICFilter(**filter_kwargs)

    async def _start_filter_with_mocks(self, filter_instance, sample_rate=16000):
        """Start a filter with mocked SDK components."""
        with (
            patch(f"{AIC_FILTER_MODULE}.Model") as mock_model_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorAsync", return_value=self.mock_processor),
        ):
            mock_model_cls.from_file.return_value = self.mock_model
            mock_model_cls.download_async = AsyncMock(return_value="/tmp/model")
            mock_config_cls.optimal.return_value = MagicMock()
            await filter_instance.start(sample_rate)

    async def test_initialization_requires_model_id_or_path(self):
        """Test filter initialization fails without model_id or model_path."""
        with patch(f"{AIC_FILTER_MODULE}.set_sdk_id"):
            with self.assertRaises(ValueError) as context:
                self.AICFilter(license_key="test-key")

        self.assertIn("model_id", str(context.exception))
        self.assertIn("model_path", str(context.exception))

    async def test_initialization_with_model_id(self):
        """Test filter initialization with model_id."""
        filter_instance = self._create_filter_with_mocks()

        self.assertEqual(filter_instance._license_key, "test-key")
        self.assertEqual(filter_instance._model_id, "test-model")
        self.assertIsNone(filter_instance._model_path)
        self.assertFalse(filter_instance._bypass)

    async def test_initialization_with_model_path(self):
        """Test filter initialization with model_path."""
        model_path = Path("/tmp/test.aicmodel")
        filter_instance = self._create_filter_with_mocks(model_id=None, model_path=model_path)

        self.assertEqual(filter_instance._model_path, model_path)
        self.assertIsNone(filter_instance._model_id)

    async def test_initialization_with_custom_download_dir(self):
        """Test filter initialization with custom model_download_dir."""
        download_dir = Path("/custom/cache")
        filter_instance = self._create_filter_with_mocks(model_download_dir=download_dir)

        self.assertEqual(filter_instance._model_download_dir, download_dir)

    async def test_start_with_model_path(self):
        """Test starting filter with a local model path."""
        model_path = Path("/tmp/test.aicmodel")
        filter_instance = self._create_filter_with_mocks(model_id=None, model_path=model_path)

        with (
            patch(f"{AIC_FILTER_MODULE}.Model") as mock_model_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorAsync", return_value=self.mock_processor),
        ):
            mock_model_cls.from_file.return_value = self.mock_model
            mock_config_cls.optimal.return_value = MagicMock()

            await filter_instance.start(16000)

            mock_model_cls.from_file.assert_called_once_with(str(model_path))
            self.assertTrue(filter_instance._aic_ready)
            self.assertEqual(filter_instance._sample_rate, 16000)
            self.assertEqual(filter_instance._frames_per_block, 160)

    async def test_start_with_model_id_downloads(self):
        """Test starting filter with model_id triggers download."""
        filter_instance = self._create_filter_with_mocks()

        with (
            patch(f"{AIC_FILTER_MODULE}.Model") as mock_model_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorAsync", return_value=self.mock_processor),
        ):
            mock_model_cls.from_file.return_value = self.mock_model
            mock_model_cls.download_async = AsyncMock(return_value="/tmp/model")
            mock_config_cls.optimal.return_value = MagicMock()

            await filter_instance.start(16000)

            mock_model_cls.download_async.assert_called_once()
            mock_model_cls.from_file.assert_called_once()
            self.assertTrue(filter_instance._aic_ready)

    async def test_start_creates_processor(self):
        """Test that start creates processor with correct config."""
        filter_instance = self._create_filter_with_mocks()

        with (
            patch(f"{AIC_FILTER_MODULE}.Model") as mock_model_cls,
            patch(f"{AIC_FILTER_MODULE}.ProcessorConfig") as mock_config_cls,
            patch(
                f"{AIC_FILTER_MODULE}.ProcessorAsync", return_value=self.mock_processor
            ) as mock_processor_cls,
        ):
            mock_model_cls.from_file.return_value = self.mock_model
            mock_model_cls.download_async = AsyncMock(return_value="/tmp/model")
            mock_config_cls.optimal.return_value = MagicMock()

            await filter_instance.start(16000)

            mock_config_cls.optimal.assert_called_once()
            mock_processor_cls.assert_called_once()
            self.assertIsNotNone(filter_instance._processor_ctx)
            self.assertIsNotNone(filter_instance._vad_ctx)

    async def test_start_applies_initial_bypass_parameter(self):
        """Test that start applies bypass parameter."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        # Check that bypass was set to 0.0 (enabled)
        bypass_params = [
            (p, v)
            for p, v in self.mock_processor.processor_ctx.parameters_set
            if p == aic_sdk.ProcessorParameter.Bypass
        ]
        self.assertTrue(len(bypass_params) > 0)
        self.assertEqual(bypass_params[-1][1], 0.0)

    async def test_stop_cleans_up_resources(self):
        """Test that stop properly cleans up resources."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        await filter_instance.stop()

        self.assertTrue(self.mock_processor.processor_ctx.reset_called)
        self.assertIsNone(filter_instance._processor)
        self.assertIsNone(filter_instance._processor_ctx)
        self.assertIsNone(filter_instance._vad_ctx)
        self.assertIsNone(filter_instance._model)
        self.assertFalse(filter_instance._aic_ready)

    async def test_stop_without_start(self):
        """Test that stop can be called safely without start."""
        filter_instance = self._create_filter_with_mocks()

        # Should not raise
        await filter_instance.stop()

    async def test_process_frame_enable(self):
        """Test processing FilterEnableFrame to enable filtering."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)
        filter_instance._bypass = True

        enable_frame = self.FilterEnableFrame(enable=True)
        await filter_instance.process_frame(enable_frame)

        self.assertFalse(filter_instance._bypass)

    async def test_process_frame_disable(self):
        """Test processing FilterEnableFrame to disable filtering."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        disable_frame = self.FilterEnableFrame(enable=False)
        await filter_instance.process_frame(disable_frame)

        self.assertTrue(filter_instance._bypass)

    async def test_filter_when_not_ready(self):
        """Test that filter returns audio unchanged when not ready."""
        filter_instance = self._create_filter_with_mocks()
        # Don't call start()

        input_audio = b"\x00\x01\x02\x03"
        output_audio = await filter_instance.filter(input_audio)

        self.assertEqual(output_audio, input_audio)

    async def test_filter_with_incomplete_frame(self):
        """Test filtering audio with incomplete frame data."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        # Create audio data for less than one frame (100 samples = 200 bytes)
        samples = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        # Should return empty bytes since no complete frame
        self.assertEqual(output_audio, b"")

    async def test_filter_with_complete_frame(self):
        """Test filtering audio with exactly one complete frame."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        # Create audio data for exactly one frame (160 samples = 320 bytes)
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        self.assertIsInstance(output_audio, bytes)
        self.assertEqual(len(output_audio), len(input_audio))

    async def test_filter_with_multiple_frames(self):
        """Test filtering audio with multiple complete frames."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        # Create audio data for 3 complete frames (480 samples = 960 bytes)
        samples = np.random.randint(-32768, 32767, size=480, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        self.assertEqual(len(output_audio), len(input_audio))

    async def test_filter_with_buffering(self):
        """Test that filter properly buffers incomplete frames."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        # First call: Send 100 samples (incomplete frame)
        samples1 = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        input_audio1 = samples1.tobytes()
        output_audio1 = await filter_instance.filter(input_audio1)

        self.assertEqual(output_audio1, b"")
        self.assertEqual(len(filter_instance._audio_buffer), 200)

        # Second call: Send 60 more samples (now we have 160 total = 1 complete frame)
        samples2 = np.random.randint(-32768, 32767, size=60, dtype=np.int16)
        input_audio2 = samples2.tobytes()
        output_audio2 = await filter_instance.filter(input_audio2)

        self.assertEqual(len(output_audio2), 320)
        self.assertEqual(len(filter_instance._audio_buffer), 0)

    async def test_filter_with_partial_buffering(self):
        """Test that filter keeps remainder in buffer after processing."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        # Send 250 samples (1 complete frame + 90 samples remainder)
        samples = np.random.randint(-32768, 32767, size=250, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        self.assertEqual(len(output_audio), 320)  # 1 frame
        self.assertEqual(len(filter_instance._audio_buffer), 180)  # 90 samples * 2 bytes

    async def test_get_vad_context_before_start(self):
        """Test that get_vad_context raises before start."""
        filter_instance = self._create_filter_with_mocks()

        with self.assertRaises(RuntimeError) as context:
            filter_instance.get_vad_context()

        self.assertIn("not initialized", str(context.exception))

    async def test_get_vad_context_after_start(self):
        """Test that get_vad_context returns context after start."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        vad_ctx = filter_instance.get_vad_context()

        self.assertEqual(vad_ctx, self.mock_processor.vad_ctx)

    async def test_create_vad_analyzer(self):
        """Test create_vad_analyzer returns analyzer with factory."""
        filter_instance = self._create_filter_with_mocks()

        analyzer = filter_instance.create_vad_analyzer()

        self.assertIsNotNone(analyzer)
        # Factory should be set
        self.assertIsNotNone(analyzer._vad_context_factory)

    async def test_create_vad_analyzer_with_params(self):
        """Test create_vad_analyzer with custom parameters."""
        filter_instance = self._create_filter_with_mocks()

        analyzer = filter_instance.create_vad_analyzer(
            speech_hold_duration=0.1,
            minimum_speech_duration=0.05,
            sensitivity=8.0,
        )

        self.assertEqual(analyzer._pending_speech_hold_duration, 0.1)
        self.assertEqual(analyzer._pending_minimum_speech_duration, 0.05)
        self.assertEqual(analyzer._pending_sensitivity, 8.0)

    async def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles."""
        filter_instance = self._create_filter_with_mocks()

        for sample_rate in [16000, 24000, 48000]:
            # Create fresh mock processor for each cycle
            self.mock_processor = MockProcessor()
            await self._start_filter_with_mocks(filter_instance, sample_rate)
            self.assertTrue(filter_instance._aic_ready)
            self.assertEqual(filter_instance._sample_rate, sample_rate)

            await filter_instance.stop()
            self.assertFalse(filter_instance._aic_ready)

    async def test_concurrent_filter_calls(self):
        """Test that concurrent filter calls are handled safely."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        async def filter_audio():
            return await filter_instance.filter(input_audio)

        tasks = [filter_audio() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIsInstance(result, bytes)

    async def test_buffer_cleared_on_stop(self):
        """Test that audio buffer is cleared when stopping."""
        filter_instance = self._create_filter_with_mocks()
        await self._start_filter_with_mocks(filter_instance)

        # Add incomplete frame to buffer
        samples = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        input_audio = samples.tobytes()
        await filter_instance.filter(input_audio)

        # Verify buffer has data
        self.assertGreater(len(filter_instance._audio_buffer), 0)

        # Stop should clear buffer
        await filter_instance.stop()
        self.assertEqual(len(filter_instance._audio_buffer), 0)

    async def test_set_sdk_id_called_on_init(self):
        """Test that set_sdk_id is called during initialization."""
        with patch(f"{AIC_FILTER_MODULE}.set_sdk_id") as mock_set_sdk_id:
            self.AICFilter(license_key="test-key", model_id="test-model")

            mock_set_sdk_id.assert_called_once_with(6)


if __name__ == "__main__":
    unittest.main()
