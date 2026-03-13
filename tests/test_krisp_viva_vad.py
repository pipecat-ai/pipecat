#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for KrispVivaVadAnalyzer."""

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
from pipecat.audio.vad.krisp_viva_vad import KrispVivaVadAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


class TestKrispVivaVadAnalyzer(unittest.TestCase):
    """Test suite for KrispVivaVadAnalyzer."""

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
        self.mock_krisp_audio.VadSessionConfig.reset_mock()
        self.mock_krisp_audio.VadFloat.reset_mock()

        # Mock ModelInfo
        self.mock_model_info = MagicMock()
        self.mock_krisp_audio.ModelInfo.return_value = self.mock_model_info

        # Mock VadSessionConfig
        self.mock_vad_cfg = MagicMock()
        self.mock_krisp_audio.VadSessionConfig.return_value = self.mock_vad_cfg

        # Mock VAD session
        self.mock_session = MagicMock()
        self.mock_session.process = MagicMock(return_value=0.75)  # Return voice probability
        self.mock_krisp_audio.VadFloat.create.return_value = self.mock_session

        # Patch krisp_audio in the module
        self.krisp_audio_patch = patch(
            "pipecat.audio.vad.krisp_viva_vad.krisp_audio", self.mock_krisp_audio
        )
        self.krisp_audio_patch.start()

        # Patch KrispVivaSDKManager
        self.sdk_manager_patcher = patch(
            "pipecat.audio.vad.krisp_viva_vad.KrispVivaSDKManager"
        )
        self.mock_sdk_manager = self.sdk_manager_patcher.start()
        self.mock_sdk_manager.acquire = MagicMock()
        self.mock_sdk_manager.release = MagicMock()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Stop all patchers
        self.krisp_audio_patch.stop()
        self.sdk_manager_patcher.stop()

        # Remove temporary model file
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    def test_initialization_with_model_path(self):
        """Test analyzer initialization with explicit model path."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)

        # Verify SDK was acquired during initialization
        self.mock_sdk_manager.acquire.assert_called_once()

        # Verify analyzer attributes
        self.assertEqual(analyzer._model_path, self.model_path)
        self.assertEqual(analyzer._frame_duration_ms, 10)  # Default frame duration
        self.assertIsNone(analyzer._session)  # Session created in set_sample_rate

    def test_initialization_with_env_variable(self):
        """Test analyzer initialization using KRISP_VIVA_VAD_MODEL_PATH environment variable."""
        with patch.dict(os.environ, {"KRISP_VIVA_VAD_MODEL_PATH": self.model_path}):
            analyzer = KrispVivaVadAnalyzer()

            self.mock_sdk_manager.acquire.assert_called_once()
            self.assertEqual(analyzer._model_path, self.model_path)

    def test_initialization_without_model_path(self):
        """Test analyzer initialization fails without model path."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                KrispVivaVadAnalyzer()

            self.assertIn("Model path", str(context.exception))
            # acquire() is not called because exception is raised before it
            self.mock_sdk_manager.acquire.assert_not_called()
            # release() is called in exception handler and also in __del__ when object is destroyed
            self.assertGreaterEqual(self.mock_sdk_manager.release.call_count, 1)

    def test_initialization_with_invalid_extension(self):
        """Test analyzer initialization fails with non-.kef file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"dummy")
            tmp_path = tmp.name

        try:
            with self.assertRaises(Exception) as context:
                KrispVivaVadAnalyzer(model_path=tmp_path)

            self.assertIn(".kef extension", str(context.exception))
            # acquire() is not called because exception is raised before it
            self.mock_sdk_manager.acquire.assert_not_called()
            # release() is called in exception handler and also in __del__ when object is destroyed
            self.assertGreaterEqual(self.mock_sdk_manager.release.call_count, 1)
        finally:
            os.unlink(tmp_path)

    def test_initialization_with_nonexistent_file(self):
        """Test analyzer initialization fails with non-existent model file."""
        with self.assertRaises(FileNotFoundError):
            KrispVivaVadAnalyzer(model_path="/nonexistent/path/model.kef")

        # acquire() is not called because exception is raised before it
        self.mock_sdk_manager.acquire.assert_not_called()
        # release() is called in exception handler and also in __del__ when object is destroyed
        self.assertGreaterEqual(self.mock_sdk_manager.release.call_count, 1)

    def test_initialization_with_custom_frame_duration(self):
        """Test analyzer initialization with custom frame duration."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path, frame_duration=20)

        self.assertEqual(analyzer._frame_duration_ms, 20)

    def test_initialization_with_sample_rate(self):
        """Test analyzer initialization with sample rate."""
        analyzer = KrispVivaVadAnalyzer(
            model_path=self.model_path, sample_rate=16000, frame_duration=10
        )

        # Should calculate samples per frame
        self.assertEqual(analyzer._samples_per_frame, 160)  # 16000 * 10 / 1000

    def test_set_sample_rate_with_supported_rates(self):
        """Test setting sample rate with supported values."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)

        for sample_rate in [8000, 16000, 32000, 44100, 48000]:
            analyzer.set_sample_rate(sample_rate)

            # Verify session was created
            self.assertIsNotNone(analyzer._session)
            self.assertEqual(analyzer.sample_rate, sample_rate)

            # Verify samples per frame was calculated
            expected_samples = int((sample_rate * analyzer._frame_duration_ms) / 1000)
            self.assertEqual(analyzer._samples_per_frame, expected_samples)

            # Verify VadSessionConfig was created and configured
            self.mock_krisp_audio.VadSessionConfig.assert_called()
            self.assertEqual(self.mock_vad_cfg.modelInfo, self.mock_model_info)

    def test_set_sample_rate_with_unsupported_rate(self):
        """Test setting sample rate with unsupported value raises ValueError."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)

        with self.assertRaises(ValueError) as context:
            analyzer.set_sample_rate(12000)  # Unsupported sample rate

        self.assertIn("sample rate needs to be", str(context.exception))

    def test_num_frames_required_with_session(self):
        """Test num_frames_required when session is created."""
        analyzer = KrispVivaVadAnalyzer(
            model_path=self.model_path, sample_rate=16000, frame_duration=10
        )
        analyzer.set_sample_rate(16000)

        # Should return samples per frame from session
        frames_required = analyzer.num_frames_required()
        self.assertEqual(frames_required, 160)  # 16000 * 10 / 1000

    def test_num_frames_required_without_session(self):
        """Test num_frames_required when session is not created yet."""
        analyzer = KrispVivaVadAnalyzer(
            model_path=self.model_path, sample_rate=16000, frame_duration=10
        )

        # Should calculate from sample_rate
        frames_required = analyzer.num_frames_required()
        self.assertEqual(frames_required, 160)

    def test_num_frames_required_with_different_sample_rates(self):
        """Test num_frames_required with different sample rates."""
        test_cases = [
            (8000, 10, 80),  # 8kHz @ 10ms = 80 samples
            (16000, 10, 160),  # 16kHz @ 10ms = 160 samples
            (16000, 20, 320),  # 16kHz @ 20ms = 320 samples
            (32000, 10, 320),  # 32kHz @ 10ms = 320 samples
            (44100, 10, 441),  # 44.1kHz @ 10ms = 441 samples
            (48000, 10, 480),  # 48kHz @ 10ms = 480 samples
        ]

        for sample_rate, frame_duration, expected_frames in test_cases:
            analyzer = KrispVivaVadAnalyzer(
                model_path=self.model_path,
                sample_rate=sample_rate,
                frame_duration=frame_duration,
            )
            analyzer.set_sample_rate(sample_rate)

            frames_required = analyzer.num_frames_required()
            self.assertEqual(
                frames_required,
                expected_frames,
                f"Failed for {sample_rate}Hz @ {frame_duration}ms",
            )

    def test_num_frames_required_fallback(self):
        """Test num_frames_required fallback when sample rate not set."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path, frame_duration=10)

        # Should use default fallback (16kHz)
        frames_required = analyzer.num_frames_required()
        self.assertEqual(frames_required, 160)  # 16000 * 10 / 1000

    def test_voice_confidence_with_valid_buffer(self):
        """Test voice_confidence with valid audio buffer."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)
        analyzer.set_sample_rate(16000)

        # Create audio buffer for one frame (160 samples = 320 bytes)
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        audio_buffer = samples.tobytes()

        confidence = analyzer.voice_confidence(audio_buffer)

        # Verify confidence is returned
        self.assertIsInstance(confidence, float)
        self.assertEqual(confidence, 0.75)  # Mock returns 0.75

        # Verify session.process was called
        self.mock_session.process.assert_called_once()

    def test_voice_confidence_without_session(self):
        """Test voice_confidence when session is not initialized."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)

        # Create audio buffer
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        audio_buffer = samples.tobytes()

        # Should return 0.0 and log warning
        confidence = analyzer.voice_confidence(audio_buffer)
        self.assertEqual(confidence, 0.0)

        # Verify session.process was NOT called
        self.mock_session.process.assert_not_called()

    def test_voice_confidence_error_handling(self):
        """Test voice_confidence handles processing errors gracefully."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)
        analyzer.set_sample_rate(16000)

        # Make session.process raise an exception
        self.mock_session.process.side_effect = Exception("Processing error")

        # Create audio buffer
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        audio_buffer = samples.tobytes()

        # Should return 0.0 on error
        confidence = analyzer.voice_confidence(audio_buffer)
        self.assertEqual(confidence, 0.0)

    def test_voice_confidence_audio_conversion(self):
        """Test voice_confidence properly converts audio format."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)
        analyzer.set_sample_rate(16000)

        # Create deterministic audio buffer
        samples = np.array([1000, -2000, 3000, -4000], dtype=np.int16)
        audio_buffer = samples.tobytes()

        analyzer.voice_confidence(audio_buffer)

        # Verify process was called with float32 array
        call_args = self.mock_session.process.call_args[0][0]
        self.assertIsInstance(call_args, np.ndarray)
        self.assertEqual(call_args.dtype, np.float32)

        # Verify normalization (int16 to float32)
        expected_float = samples.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(call_args, expected_float)

    def test_voice_confidence_different_buffer_sizes(self):
        """Test voice_confidence with different buffer sizes."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)
        analyzer.set_sample_rate(16000)

        test_sizes = [80, 160, 320, 480]  # Different numbers of samples

        for size in test_sizes:
            samples = np.random.randint(-32768, 32767, size=size, dtype=np.int16)
            audio_buffer = samples.tobytes()

            confidence = analyzer.voice_confidence(audio_buffer)

            # Should always return a float
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_cleanup_on_destruction(self):
        """Test that cleanup happens when analyzer is destroyed."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)
        analyzer.set_sample_rate(16000)

        # Reset mock to track calls
        self.mock_sdk_manager.release.reset_mock()

        # Delete analyzer (triggers __del__)
        del analyzer

        # Verify SDK was released
        self.mock_sdk_manager.release.assert_called_once()

    def test_initialization_with_vad_params(self):
        """Test analyzer initialization with VAD parameters."""
        params = VADParams(confidence=0.8, start_secs=0.3, stop_secs=0.9)
        analyzer = KrispVivaVadAnalyzer(
            model_path=self.model_path, sample_rate=16000, params=params
        )

        self.assertEqual(analyzer.params.confidence, 0.8)
        self.assertEqual(analyzer.params.start_secs, 0.3)
        self.assertEqual(analyzer.params.stop_secs, 0.9)

    def test_set_sample_rate_creates_session(self):
        """Test that set_sample_rate creates a new session."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)

        # Initially no session
        self.assertIsNone(analyzer._session)

        # Set sample rate should create session
        analyzer.set_sample_rate(16000)

        # Verify session was created
        self.assertIsNotNone(analyzer._session)
        self.mock_krisp_audio.VadFloat.create.assert_called_once()

    def test_set_sample_rate_replaces_session(self):
        """Test that set_sample_rate replaces existing session."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)

        # Create first session
        analyzer.set_sample_rate(16000)
        session1 = analyzer._session
        self.assertIsNotNone(session1)

        # Reset mock to track new calls
        self.mock_krisp_audio.VadFloat.create.reset_mock()

        # Set different sample rate should create new session
        analyzer.set_sample_rate(48000)
        session2 = analyzer._session
        self.assertIsNotNone(session2)

        # Verify new session was created
        self.mock_krisp_audio.VadFloat.create.assert_called_once()

    def test_session_configuration(self):
        """Test that session is configured correctly."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path, frame_duration=15)
        analyzer.set_sample_rate(16000)

        # Verify VadSessionConfig was configured
        self.assertEqual(self.mock_vad_cfg.modelInfo, self.mock_model_info)
        self.assertEqual(self.mock_vad_cfg.inputFrameDuration, "15ms")

        # Verify sample rate was set
        from pipecat.audio.krisp_instance import int_to_krisp_sample_rate

        expected_sample_rate = int_to_krisp_sample_rate(16000)
        self.assertEqual(self.mock_vad_cfg.inputSampleRate, expected_sample_rate)

    def test_multiple_sample_rate_changes(self):
        """Test multiple sample rate changes."""
        analyzer = KrispVivaVadAnalyzer(model_path=self.model_path)

        sample_rates = [8000, 16000, 32000, 48000]

        for sample_rate in sample_rates:
            analyzer.set_sample_rate(sample_rate)
            self.assertEqual(analyzer.sample_rate, sample_rate)

            # Verify num_frames_required is correct
            expected_frames = int((sample_rate * 10) / 1000)
            self.assertEqual(analyzer.num_frames_required(), expected_frames)


if __name__ == "__main__":
    unittest.main()
