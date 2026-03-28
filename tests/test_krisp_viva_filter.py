#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Mock package version check before importing pipecat
# This allows tests to run in development mode without installed package
_version_patcher = patch("importlib.metadata.version", return_value="0.0.0-dev")
_version_patcher.start()

# Mock krisp_audio module BEFORE any pipecat imports
# This allows tests to run without krisp_audio installed
mock_krisp_audio = MagicMock()
mock_krisp_audio.SamplingRate.Sr8000Hz = 8000
mock_krisp_audio.SamplingRate.Sr16000Hz = 16000
mock_krisp_audio.SamplingRate.Sr24000Hz = 24000
mock_krisp_audio.SamplingRate.Sr32000Hz = 32000
mock_krisp_audio.SamplingRate.Sr44100Hz = 44100
mock_krisp_audio.SamplingRate.Sr48000Hz = 48000
mock_krisp_audio.FrameDuration.Fd10ms = "10ms"
mock_krisp_audio.FrameDuration.Fd15ms = "15ms"
mock_krisp_audio.FrameDuration.Fd20ms = "20ms"
mock_krisp_audio.FrameDuration.Fd30ms = "30ms"
mock_krisp_audio.FrameDuration.Fd32ms = "32ms"

# Install the mock in sys.modules before importing
sys.modules["krisp_audio"] = mock_krisp_audio

# Mock pipecat_ai_krisp package
mock_pipecat_krisp = MagicMock()
sys.modules["pipecat_ai_krisp"] = mock_pipecat_krisp
sys.modules["pipecat_ai_krisp.audio"] = MagicMock()
sys.modules["pipecat_ai_krisp.audio.krisp_processor"] = MagicMock()

# Now we can safely import
from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
from pipecat.frames.frames import FilterEnableFrame


class TestKrispVivaFilter(unittest.IsolatedAsyncioTestCase):
    """Test suite for KrispVivaFilter audio filter."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary .kef model file for testing
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix=".kef", delete=False)
        self.temp_model_file.write(b"dummy model data")
        self.temp_model_file.close()
        self.model_path = self.temp_model_file.name

        # Use the global mock_krisp_audio that was set up before imports
        self.mock_krisp_audio = mock_krisp_audio

        # Reset all mocks to clear call counts from previous tests
        self.mock_krisp_audio.reset_mock()
        self.mock_krisp_audio.ModelInfo.reset_mock()
        self.mock_krisp_audio.NcSessionConfig.reset_mock()
        self.mock_krisp_audio.NcInt16.reset_mock()

        # Mock ModelInfo
        self.mock_model_info = MagicMock()
        self.mock_krisp_audio.ModelInfo.return_value = self.mock_model_info

        # Mock NcSessionConfig
        self.mock_nc_cfg = MagicMock()
        self.mock_krisp_audio.NcSessionConfig.return_value = self.mock_nc_cfg

        # Mock session
        self.mock_session = MagicMock()
        self.mock_session.process = MagicMock(side_effect=lambda x, level: x)
        self.mock_krisp_audio.NcInt16.create.return_value = self.mock_session

        # Patch krisp_audio in the module
        self.sample_rates_patch = patch(
            "pipecat.audio.filters.krisp_viva_filter.krisp_audio", self.mock_krisp_audio
        )
        self.sample_rates_patch.start()

        # Patch KrispVivaSDKManager
        self.sdk_manager_patcher = patch(
            "pipecat.audio.filters.krisp_viva_filter.KrispVivaSDKManager"
        )
        self.mock_sdk_manager = self.sdk_manager_patcher.start()
        self.mock_sdk_manager.acquire = MagicMock()
        self.mock_sdk_manager.release = MagicMock()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Stop all patchers
        self.sample_rates_patch.stop()
        self.sdk_manager_patcher.stop()

        # Remove temporary model file
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    async def test_initialization_with_model_path(self):
        """Test filter initialization with explicit model path."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        # Verify SDK was NOT acquired during initialization (happens in start())
        self.mock_sdk_manager.acquire.assert_not_called()

        # Verify filter attributes
        self.assertEqual(filter_instance._model_path, self.model_path)
        self.assertTrue(filter_instance._filtering)  # Filtering starts enabled
        self.assertEqual(filter_instance._noise_suppression_level, 100)
        self.assertIsNotNone(filter_instance._audio_buffer)

    async def test_initialization_with_env_variable(self):
        """Test filter initialization using KRISP_VIVA_FILTER_MODEL_PATH environment variable."""
        with patch.dict(os.environ, {"KRISP_VIVA_FILTER_MODEL_PATH": self.model_path}):
            filter_instance = KrispVivaFilter()

            # Verify SDK was NOT acquired during initialization (happens in start())
            self.mock_sdk_manager.acquire.assert_not_called()
            self.assertEqual(filter_instance._model_path, self.model_path)

    async def test_initialization_without_model_path(self):
        """Test filter initialization fails without model path."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                KrispVivaFilter()

            self.assertIn("Model path", str(context.exception))
            # SDK acquire not called during initialization (happens in start())
            # But release() is called in exception handler even though acquire() wasn't called
            self.mock_sdk_manager.acquire.assert_not_called()
            self.mock_sdk_manager.release.assert_called_once()

    async def test_initialization_with_invalid_extension(self):
        """Test filter initialization fails with non-.kef file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"dummy")
            tmp_path = tmp.name

        try:
            with self.assertRaises(Exception) as context:
                KrispVivaFilter(model_path=tmp_path)

            self.assertIn(".kef extension", str(context.exception))
            # SDK acquire not called during initialization (happens in start())
            # But release() is called in exception handler even though acquire() wasn't called
            self.mock_sdk_manager.acquire.assert_not_called()
            self.mock_sdk_manager.release.assert_called_once()
        finally:
            os.unlink(tmp_path)

    async def test_initialization_with_nonexistent_file(self):
        """Test filter initialization fails with non-existent model file."""
        with self.assertRaises(FileNotFoundError):
            KrispVivaFilter(model_path="/nonexistent/path/model.kef")

        # SDK acquire not called during initialization (happens in start())
        # But release() is called in exception handler even though acquire() wasn't called
        self.mock_sdk_manager.acquire.assert_not_called()
        self.mock_sdk_manager.release.assert_called_once()

    async def test_initialization_with_custom_noise_level(self):
        """Test filter initialization with custom noise suppression level."""
        filter_instance = KrispVivaFilter(model_path=self.model_path, noise_suppression_level=50)

        self.assertEqual(filter_instance._noise_suppression_level, 50)

    async def test_initialization_with_default_noise_level(self):
        """Test filter initialization with default noise suppression level."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        self.assertEqual(filter_instance._noise_suppression_level, 100)

    async def test_start_with_supported_sample_rate(self):
        """Test starting filter with a supported sample rate."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        await filter_instance.start(16000)

        # Verify SDK was acquired during start()
        self.mock_sdk_manager.acquire.assert_called_once()

        # Verify session was created
        self.assertIsNotNone(filter_instance._session)
        self.assertEqual(filter_instance._current_sample_rate, 16000)
        self.assertEqual(filter_instance._samples_per_frame, 160)  # 16000 * 10ms / 1000

        # Verify NcSessionConfig was created and configured
        # Note: Called once in start() (no preload session anymore)
        self.assertEqual(self.mock_krisp_audio.NcSessionConfig.call_count, 1)
        # Verify frame duration was set (hardcoded to 10ms in filter)
        self.assertEqual(self.mock_nc_cfg.inputFrameDuration, "10ms")
        # inputSampleRate and outputSampleRate are now set to the enum value
        from pipecat.audio.krisp_instance import int_to_krisp_sample_rate

        expected_sample_rate = int_to_krisp_sample_rate(16000)
        self.assertEqual(self.mock_nc_cfg.inputSampleRate, expected_sample_rate)
        self.assertEqual(self.mock_nc_cfg.outputSampleRate, expected_sample_rate)

    async def test_start_with_unsupported_sample_rate(self):
        """Test starting filter with an unsupported sample rate raises RuntimeError."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        with self.assertRaises(RuntimeError) as context:
            await filter_instance.start(12000)  # Unsupported sample rate

        self.assertIn("Unsupported sample rate", str(context.exception))

    async def test_start_multiple_sample_rates(self):
        """Test starting filter with multiple different sample rates."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        for sample_rate in [8000, 16000, 24000, 32000, 44100, 48000]:
            # Reset mock config for each iteration to verify frame duration is always set
            mock_nc_cfg = MagicMock()
            self.mock_krisp_audio.NcSessionConfig.return_value = mock_nc_cfg

            await filter_instance.start(sample_rate)
            self.assertEqual(filter_instance._current_sample_rate, sample_rate)
            expected_samples = int((sample_rate * 10) / 1000)
            self.assertEqual(filter_instance._samples_per_frame, expected_samples)

            # Verify frame duration is always set to 10ms (hardcoded in filter)
            self.assertEqual(mock_nc_cfg.inputFrameDuration, "10ms")

    async def test_stop(self):
        """Test stopping the filter."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        await filter_instance.stop()

        # Verify session was cleared
        self.assertIsNone(filter_instance._session)

    async def test_process_frame_enable(self):
        """Test processing FilterEnableFrame to enable filtering."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        # Disable filtering first
        filter_instance._filtering = False

        enable_frame = FilterEnableFrame(enable=True)
        await filter_instance.process_frame(enable_frame)

        self.assertTrue(filter_instance._filtering)

    async def test_process_frame_disable(self):
        """Test processing FilterEnableFrame to disable filtering."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)
        # After start, filtering should be enabled
        self.assertTrue(filter_instance._filtering)

        disable_frame = FilterEnableFrame(enable=False)
        await filter_instance.process_frame(disable_frame)

        self.assertFalse(filter_instance._filtering)

    async def test_filter_when_disabled(self):
        """Test that filter returns audio unchanged when filtering is disabled."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)
        # Disable filtering
        filter_instance._filtering = False

        input_audio = b"\x00\x01\x02\x03\x04\x05"
        output_audio = await filter_instance.filter(input_audio)

        self.assertEqual(output_audio, input_audio)

    async def test_filter_with_complete_frame(self):
        """Test filtering audio with exactly one complete frame."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Create audio data for exactly one 10ms frame (160 samples = 320 bytes)
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        # Verify audio was processed
        self.assertIsInstance(output_audio, bytes)
        self.assertEqual(len(output_audio), len(input_audio))

        # Verify session.process was called
        self.mock_session.process.assert_called()

    async def test_filter_with_multiple_frames(self):
        """Test filtering audio with multiple complete frames."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Create audio data for 3 complete 10ms frames (480 samples = 960 bytes)
        samples = np.random.randint(-32768, 32767, size=480, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        # Verify audio was processed
        self.assertIsInstance(output_audio, bytes)
        self.assertEqual(len(output_audio), len(input_audio))

        # Verify session.process was called 3 times
        self.assertEqual(self.mock_session.process.call_count, 3)

    async def test_filter_with_incomplete_frame(self):
        """Test filtering audio with incomplete frame data."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Create audio data for less than one frame (100 samples = 200 bytes)
        samples = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        # Should return empty bytes since no complete frame
        self.assertEqual(output_audio, b"")

        # Verify session.process was NOT called
        self.mock_session.process.assert_not_called()

    async def test_filter_with_buffering(self):
        """Test that filter properly buffers incomplete frames."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # First call: Send 100 samples (incomplete frame)
        samples1 = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        input_audio1 = samples1.tobytes()
        output_audio1 = await filter_instance.filter(input_audio1)

        # Should buffer and return empty
        self.assertEqual(output_audio1, b"")
        self.assertEqual(len(filter_instance._audio_buffer), 200)

        # Second call: Send 60 more samples (now we have 160 total = 1 complete frame)
        samples2 = np.random.randint(-32768, 32767, size=60, dtype=np.int16)
        input_audio2 = samples2.tobytes()
        output_audio2 = await filter_instance.filter(input_audio2)

        # Should process one frame and return 320 bytes
        self.assertEqual(len(output_audio2), 320)
        self.assertEqual(len(filter_instance._audio_buffer), 0)
        self.mock_session.process.assert_called_once()

    async def test_filter_with_partial_buffering(self):
        """Test that filter keeps remainder in buffer after processing."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Send 250 samples (1 complete frame + 90 samples remainder)
        samples = np.random.randint(-32768, 32767, size=250, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        # Should process one frame (320 bytes)
        self.assertEqual(len(output_audio), 320)

        # Should keep remainder (90 samples = 180 bytes) in buffer
        self.assertEqual(len(filter_instance._audio_buffer), 180)

        self.mock_session.process.assert_called_once()

    async def test_filter_error_handling(self):
        """Test that filter handles processing errors gracefully."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Make session.process raise an exception
        self.mock_session.process.side_effect = Exception("Processing error")

        # Create audio data for one complete frame
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        # Should return original audio on error
        output_audio = await filter_instance.filter(input_audio)
        self.assertEqual(output_audio, input_audio)

    async def test_filter_different_sample_rates(self):
        """Test filtering with different sample rates."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        test_cases = [
            (8000, 80),  # 8kHz: 80 samples per 10ms frame
            (16000, 160),  # 16kHz: 160 samples per 10ms frame
            (48000, 480),  # 48kHz: 480 samples per 10ms frame
        ]

        for sample_rate, expected_samples in test_cases:
            await filter_instance.start(sample_rate)

            # Create audio data for exactly one frame
            samples = np.random.randint(-32768, 32767, size=expected_samples, dtype=np.int16)
            input_audio = samples.tobytes()

            output_audio = await filter_instance.filter(input_audio)

            # Verify correct processing
            self.assertEqual(len(output_audio), len(input_audio))

    async def test_stop_releases_sdk(self):
        """Test that stop() properly releases SDK reference."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Stop the filter
        await filter_instance.stop()

        # Verify SDK was released
        self.mock_sdk_manager.release.assert_called_once()

    async def test_int_to_sample_rate_conversion(self):
        """Test sample rate conversion using the shared utility function."""
        from pipecat.audio.krisp_instance import KRISP_SAMPLE_RATES, int_to_krisp_sample_rate

        # Test valid sample rates - verify they return the correct enum values
        for rate in [8000, 16000, 24000, 32000, 44100, 48000]:
            result = int_to_krisp_sample_rate(rate)
            # Check that result is from the KRISP_SAMPLE_RATES dict
            self.assertEqual(result, KRISP_SAMPLE_RATES[rate])

        # Test invalid sample rate
        with self.assertRaises(ValueError) as context:
            int_to_krisp_sample_rate(12000)

        self.assertIn("Unsupported sample rate", str(context.exception))

    async def test_noise_suppression_level_applied(self):
        """Test that noise suppression level is passed to processing."""
        filter_instance = KrispVivaFilter(model_path=self.model_path, noise_suppression_level=75)
        await filter_instance.start(16000)

        # Create audio data for one frame
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        await filter_instance.filter(input_audio)

        # Verify noise suppression level was passed to process()
        call_args = self.mock_session.process.call_args
        self.assertEqual(call_args[0][1], 75)  # Second argument should be the level

    async def test_start_acquires_sdk(self):
        """Test that start() acquires SDK reference and creates session."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        # Verify no session exists before start
        self.assertIsNone(filter_instance._session)

        # Start the filter
        await filter_instance.start(16000)

        # Verify SDK was acquired
        self.mock_sdk_manager.acquire.assert_called_once()

        # Verify session was created
        self.assertIsNotNone(filter_instance._session)

        # Verify NcSessionConfig was created and frame duration was set
        self.mock_krisp_audio.NcSessionConfig.assert_called_once()
        # Verify frame duration was set to 10ms (hardcoded in filter)
        self.assertEqual(self.mock_nc_cfg.inputFrameDuration, "10ms")

    async def test_filter_preserves_audio_data_integrity(self):
        """Test that filter processing preserves data integrity."""
        # Make mock session return the same data
        self.mock_session.process.side_effect = lambda x, level: x.copy()

        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Create deterministic audio data
        samples = np.arange(160, dtype=np.int16)
        input_audio = samples.tobytes()

        output_audio = await filter_instance.filter(input_audio)

        # Verify output matches input (since mock returns same data)
        output_samples = np.frombuffer(output_audio, dtype=np.int16)
        np.testing.assert_array_equal(output_samples, samples)

    # ==================== Concurrency & Thread Safety Tests ====================

    async def test_concurrent_filter_calls(self):
        """Test that concurrent filter calls are handled safely."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Create audio data for one frame
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        # Create multiple concurrent filter calls
        async def filter_audio():
            return await filter_instance.filter(input_audio)

        # Run 10 concurrent filter operations
        tasks = [filter_audio() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all calls completed successfully
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIsInstance(result, bytes)
            self.assertEqual(len(result), len(input_audio))

        # Verify session.process was called for each frame
        self.assertEqual(self.mock_session.process.call_count, 10)

    async def test_concurrent_enable_disable(self):
        """Test rapid enable/disable toggling during filtering."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Create audio data
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        # Concurrently toggle enable/disable while filtering
        async def toggle_and_filter(toggle_enable):
            enable_frame = FilterEnableFrame(enable=toggle_enable)
            await filter_instance.process_frame(enable_frame)
            return await filter_instance.filter(input_audio)

        # Run concurrent enable/disable operations
        tasks = [
            toggle_and_filter(True),
            toggle_and_filter(False),
            toggle_and_filter(True),
            toggle_and_filter(False),
        ]
        results = await asyncio.gather(*tasks)

        # Verify all operations completed
        self.assertEqual(len(results), 4)

        # Verify final state is consistent (last operation was disable)
        self.assertFalse(filter_instance._filtering)

    async def test_concurrent_start_stop(self):
        """Test concurrent start/stop operations."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        async def start_filter():
            await filter_instance.start(16000)

        async def stop_filter():
            await filter_instance.stop()

        # Run start and stop concurrently
        await asyncio.gather(start_filter(), stop_filter())

        # Verify final state (stop should clear session)
        # Note: This tests that operations don't crash, final state may vary
        # depending on which completes first

    async def test_concurrent_filter_with_state_changes(self):
        """Test filtering while state changes occur concurrently."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        async def filter_operation():
            return await filter_instance.filter(input_audio)

        async def toggle_filtering():
            # Toggle based on current filtering state
            is_filtering = filter_instance._filtering
            enable_frame = FilterEnableFrame(enable=not is_filtering)
            await filter_instance.process_frame(enable_frame)

        # Run filtering and toggling concurrently
        filter_tasks = [filter_operation() for _ in range(5)]
        toggle_tasks = [toggle_filtering() for _ in range(3)]

        results = await asyncio.gather(*filter_tasks + toggle_tasks)

        # Verify all operations completed without errors
        self.assertEqual(len(results), 8)

    # ==================== State Transition Tests ====================

    async def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        # First cycle
        await filter_instance.start(16000)
        self.assertIsNotNone(filter_instance._session)
        self.assertEqual(filter_instance._current_sample_rate, 16000)

        await filter_instance.stop()
        self.assertIsNone(filter_instance._session)

        # Second cycle
        await filter_instance.start(24000)
        self.assertIsNotNone(filter_instance._session)
        self.assertEqual(filter_instance._current_sample_rate, 24000)

        await filter_instance.stop()
        self.assertIsNone(filter_instance._session)

        # Third cycle
        await filter_instance.start(48000)
        self.assertIsNotNone(filter_instance._session)
        self.assertEqual(filter_instance._current_sample_rate, 48000)

        await filter_instance.stop()
        self.assertIsNone(filter_instance._session)

        # Verify session was created multiple times
        self.assertGreaterEqual(self.mock_krisp_audio.NcInt16.create.call_count, 3)

    async def test_sample_rate_change_during_operation(self):
        """Test changing sample rate between start/stop cycles."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        # Start with 16kHz
        await filter_instance.start(16000)
        self.assertEqual(filter_instance._current_sample_rate, 16000)
        self.assertEqual(filter_instance._samples_per_frame, 160)

        # Process some audio
        samples_16k = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        output_16k = await filter_instance.filter(samples_16k.tobytes())
        self.assertEqual(len(output_16k), 320)  # 160 samples * 2 bytes

        # Stop and change to 48kHz
        await filter_instance.stop()
        await filter_instance.start(48000)
        self.assertEqual(filter_instance._current_sample_rate, 48000)
        self.assertEqual(filter_instance._samples_per_frame, 480)

        # Process audio at new sample rate
        samples_48k = np.random.randint(-32768, 32767, size=480, dtype=np.int16)
        output_48k = await filter_instance.filter(samples_48k.tobytes())
        self.assertEqual(len(output_48k), 960)  # 480 samples * 2 bytes

        await filter_instance.stop()

    async def test_start_after_stop_with_different_sample_rate(self):
        """Test starting with different sample rate after stop."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        # Start with 8kHz
        await filter_instance.start(8000)
        self.assertEqual(filter_instance._current_sample_rate, 8000)
        await filter_instance.stop()

        # Start with 32kHz
        await filter_instance.start(32000)
        self.assertEqual(filter_instance._current_sample_rate, 32000)
        await filter_instance.stop()

        # Start with 44.1kHz
        await filter_instance.start(44100)
        self.assertEqual(filter_instance._current_sample_rate, 44100)
        await filter_instance.stop()

    async def test_filter_state_persistence_across_start_stop(self):
        """Test that filtering state persists across start/stop cycles."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        # Filter starts with filtering enabled
        self.assertTrue(filter_instance._filtering)

        # Start the filter
        await filter_instance.start(16000)
        self.assertTrue(filter_instance._filtering)
        self.assertIsNotNone(filter_instance._session)

        # Disable filtering
        disable_frame = FilterEnableFrame(enable=False)
        await filter_instance.process_frame(disable_frame)
        self.assertFalse(filter_instance._filtering)

        # Stop the filter (cleanup)
        await filter_instance.stop()
        self.assertIsNone(filter_instance._session)

        # Enable filtering again
        enable_frame = FilterEnableFrame(enable=True)
        await filter_instance.process_frame(enable_frame)
        self.assertTrue(filter_instance._filtering)

        # Start the filter again
        await filter_instance.start(16000)
        self.assertTrue(filter_instance._filtering)
        self.assertIsNotNone(filter_instance._session)

    async def test_noise_suppression_level_persistence(self):
        """Test that noise suppression level persists across start/stop."""
        filter_instance = KrispVivaFilter(model_path=self.model_path, noise_suppression_level=75)

        self.assertEqual(filter_instance._noise_suppression_level, 75)

        # Start and stop
        await filter_instance.start(16000)
        await filter_instance.stop()

        # Verify noise suppression level persisted
        self.assertEqual(filter_instance._noise_suppression_level, 75)

    async def test_buffer_cleared_on_stop(self):
        """Test that audio buffer is cleared when stopping."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        await filter_instance.start(16000)

        # Add incomplete frame to buffer
        samples = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        input_audio = samples.tobytes()
        await filter_instance.filter(input_audio)

        # Verify buffer has data
        self.assertGreater(len(filter_instance._audio_buffer), 0)

        # Stop should clear buffer (or at least not cause issues)
        await filter_instance.stop()

        # Buffer state after stop - verify no errors on next start
        await filter_instance.start(16000)
        # Should be able to filter after restart
        output = await filter_instance.filter(input_audio)
        self.assertIsInstance(output, bytes)

    async def test_multiple_starts_without_stop(self):
        """Test behavior when start is called multiple times without stop."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        # First start
        await filter_instance.start(16000)
        session1 = filter_instance._session
        self.assertIsNotNone(session1)

        # Second start without stop (should replace session)
        await filter_instance.start(24000)
        session2 = filter_instance._session
        self.assertIsNotNone(session2)
        self.assertEqual(filter_instance._current_sample_rate, 24000)

        # Third start
        await filter_instance.start(48000)
        session3 = filter_instance._session
        self.assertIsNotNone(session3)
        self.assertEqual(filter_instance._current_sample_rate, 48000)

        await filter_instance.stop()

    async def test_stop_without_start(self):
        """Test that stop can be called safely without start."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        # Stop without starting should not raise an error
        await filter_instance.stop()

        # Verify session is None
        self.assertIsNone(filter_instance._session)

        # Should be able to start after stop without start
        await filter_instance.start(16000)
        self.assertIsNotNone(filter_instance._session)


if __name__ == "__main__":
    unittest.main()
