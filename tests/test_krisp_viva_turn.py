#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Check if krisp_audio is available - skip tests if not
# Note: krisp_audio is a private/proprietary module, so these tests are skipped
# in CI/CD environments where it's not available. Developers with access to
# krisp_audio can run these tests locally.
try:
    import krisp_audio

    KRISP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KRISP_AVAILABLE = False

try:
    from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
    from pipecat.audio.turn.krisp_viva_turn import (
        KrispTurnParams,
        KrispVivaTurn,
    )
except (ImportError, Exception) as e:
    # If import fails due to missing krisp_audio, mark as unavailable
    if "krisp_audio" in str(e) or "Missing module" in str(e):
        KRISP_AVAILABLE = False
        KrispVivaTurn = None
        KrispTurnParams = None
        EndOfTurnState = None
    else:
        raise


@unittest.skipIf(not KRISP_AVAILABLE, "krisp_audio module not available (private dependency)")
class TestKrispVivaTurn(unittest.IsolatedAsyncioTestCase):
    """Test suite for KrispVivaTurn turn analyzer."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary .kef model file for testing
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix=".kef", delete=False)
        self.temp_model_file.write(b"dummy model data")
        self.temp_model_file.close()
        self.model_path = self.temp_model_file.name

        # Mock krisp_audio module and its components
        self.mock_krisp_audio = MagicMock()
        self.mock_krisp_audio.SamplingRate.Sr8000Hz = 8000
        self.mock_krisp_audio.SamplingRate.Sr16000Hz = 16000
        self.mock_krisp_audio.SamplingRate.Sr24000Hz = 24000
        self.mock_krisp_audio.SamplingRate.Sr32000Hz = 32000
        self.mock_krisp_audio.SamplingRate.Sr44100Hz = 44100
        self.mock_krisp_audio.SamplingRate.Sr48000Hz = 48000

        self.mock_krisp_audio.FrameDuration.Fd10ms = "10ms"
        self.mock_krisp_audio.FrameDuration.Fd15ms = "15ms"
        self.mock_krisp_audio.FrameDuration.Fd20ms = "20ms"
        self.mock_krisp_audio.FrameDuration.Fd30ms = "30ms"
        self.mock_krisp_audio.FrameDuration.Fd32ms = "32ms"

        # Mock ModelInfo
        self.mock_model_info = MagicMock()
        self.mock_krisp_audio.ModelInfo.return_value = self.mock_model_info

        # Mock TtSessionConfig
        self.mock_tt_cfg = MagicMock()
        self.mock_krisp_audio.TtSessionConfig.return_value = self.mock_tt_cfg

        # Mock TtFloat session
        self.mock_tt_session = MagicMock()
        self.mock_tt_session.process = MagicMock(
            return_value=0.3
        )  # Default probability below threshold
        self.mock_krisp_audio.TtFloat.create.return_value = self.mock_tt_session

        # Patch krisp_audio at module level
        self.krisp_audio_patcher = patch.dict("sys.modules", {"krisp_audio": self.mock_krisp_audio})
        self.krisp_audio_patcher.start()

        # Patch the SAMPLE_RATES after import
        self.sample_rates_patch = patch(
            "pipecat.audio.turn.krisp_viva_turn.krisp_audio", self.mock_krisp_audio
        )
        self.sample_rates_patch.start()

        # Patch KrispVivaSDKManager
        self.sdk_manager_patcher = patch("pipecat.audio.turn.krisp_viva_turn.KrispVivaSDKManager")
        self.mock_sdk_manager = self.sdk_manager_patcher.start()
        self.mock_sdk_manager.acquire = MagicMock()
        self.mock_sdk_manager.release = MagicMock()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Stop all patchers
        self.krisp_audio_patcher.stop()
        self.sample_rates_patch.stop()
        self.sdk_manager_patcher.stop()

        # Remove temporary model file
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    async def test_initialization_with_model_path(self):
        """Test turn analyzer initialization with explicit model path."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)

        # Verify SDK was acquired
        self.mock_sdk_manager.acquire.assert_called_once()

        # Verify analyzer attributes
        self.assertEqual(turn_analyzer._model_path, self.model_path)
        self.assertFalse(turn_analyzer._speech_triggered)
        self.assertIsNone(turn_analyzer._speech_start_time)
        self.assertIsNone(turn_analyzer._eot_start_time)
        self.assertIsNotNone(turn_analyzer._tt_session)
        self.assertIsNotNone(turn_analyzer._preload_tt_session)

    async def test_initialization_with_env_variable(self):
        """Test turn analyzer initialization using KRISP_VIVA_TURN_MODEL_PATH environment variable."""
        with patch.dict(os.environ, {"KRISP_VIVA_TURN_MODEL_PATH": self.model_path}):
            turn_analyzer = KrispVivaTurn()

            self.mock_sdk_manager.acquire.assert_called_once()
            self.assertEqual(turn_analyzer._model_path, self.model_path)

    async def test_initialization_without_model_path(self):
        """Test turn analyzer initialization fails without model path."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                KrispVivaTurn()

            self.assertIn("Model path", str(context.exception))
            # Verify SDK was released on error
            self.mock_sdk_manager.release.assert_called_once()

    async def test_initialization_with_invalid_extension(self):
        """Test turn analyzer initialization fails with non-.kef file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"dummy")
            tmp_path = tmp.name

        try:
            with self.assertRaises(Exception) as context:
                KrispVivaTurn(model_path=tmp_path)

            self.assertIn(".kef extension", str(context.exception))
            self.mock_sdk_manager.release.assert_called_once()
        finally:
            os.unlink(tmp_path)

    async def test_initialization_with_nonexistent_file(self):
        """Test turn analyzer initialization fails with non-existent model file."""
        with self.assertRaises(FileNotFoundError):
            KrispVivaTurn(model_path="/nonexistent/path/model.kef")

        self.mock_sdk_manager.release.assert_called_once()

    async def test_initialization_with_custom_params(self):
        """Test turn analyzer initialization with custom parameters."""
        custom_params = KrispTurnParams(threshold=0.7, frame_duration_ms=30)
        turn_analyzer = KrispVivaTurn(model_path=self.model_path, params=custom_params)

        self.assertEqual(turn_analyzer._params.threshold, 0.7)
        self.assertEqual(turn_analyzer._params.frame_duration_ms, 30)

    async def test_initialization_with_default_params(self):
        """Test turn analyzer initialization with default parameters."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        self.assertEqual(turn_analyzer._params.threshold, 0.5)
        self.assertEqual(turn_analyzer._params.frame_duration_ms, 20)

    async def test_set_sample_rate_creates_session(self):
        """Test that set_sample_rate creates a turn detection session."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)

        turn_analyzer.set_sample_rate(16000)

        # Verify session was created
        self.assertIsNotNone(turn_analyzer._tt_session)
        self.assertEqual(turn_analyzer.sample_rate, 16000)
        # TtFloat.create is called twice: once for preload, once for set_sample_rate
        self.assertEqual(self.mock_krisp_audio.TtFloat.create.call_count, 2)

        # Verify TtSessionConfig was created and configured
        # Called twice: once for preload, once for set_sample_rate
        self.assertEqual(self.mock_krisp_audio.TtSessionConfig.call_count, 2)
        # Verify frame duration was set (default is 20ms)
        # Now we need to check for the enum value returned by the function
        from pipecat.audio.krisp_instance import int_to_krisp_sample_rate

        expected_sample_rate = int_to_krisp_sample_rate(16000)
        self.assertEqual(self.mock_tt_cfg.inputSampleRate, expected_sample_rate)

    async def test_set_sample_rate_with_unsupported_rate(self):
        """Test set_sample_rate with unsupported sample rate logs error and sets session to None."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)

        # This will log an error but not raise an exception
        turn_analyzer.set_sample_rate(12000)  # Unsupported sample rate

        # Verify the session is None due to error
        self.assertIsNone(turn_analyzer._tt_session)

    async def test_set_sample_rate_with_unsupported_frame_duration(self):
        """Test set_sample_rate with unsupported frame duration logs error and sets session to None."""
        turn_analyzer = KrispVivaTurn(
            model_path=self.model_path, params=KrispTurnParams(frame_duration_ms=25)
        )

        # This will log an error but not raise an exception
        turn_analyzer.set_sample_rate(16000)

        # Verify the session is None due to error
        self.assertIsNone(turn_analyzer._tt_session)

    async def test_set_sample_rate_with_different_frame_durations(self):
        """Test that different frame durations are correctly set in session config."""
        # Map frame durations to the actual enum values from FRAME_DURATIONS
        from pipecat.audio.krisp_instance import (
            int_to_krisp_frame_duration,
            int_to_krisp_sample_rate,
        )
        from pipecat.audio.turn.krisp_viva_turn import KrispVivaTurn

        # Get the KRISP_FRAME_DURATIONS mapping from the class
        frame_durations_to_test = [10, 15, 20, 30, 32]

        for duration_ms in frame_durations_to_test:
            # Create a new mock config for each iteration to avoid state pollution
            mock_tt_cfg = MagicMock()
            self.mock_krisp_audio.TtSessionConfig.return_value = mock_tt_cfg

            params = KrispTurnParams(frame_duration_ms=duration_ms)
            turn_analyzer = KrispVivaTurn(model_path=self.model_path, params=params)

            turn_analyzer.set_sample_rate(16000)

            # Verify the correct frame duration enum was set
            expected_frame_duration = int_to_krisp_frame_duration(duration_ms)
            self.assertEqual(mock_tt_cfg.inputFrameDuration, expected_frame_duration)
            # inputSampleRate is now set to the enum value
            expected_sample_rate = int_to_krisp_sample_rate(16000)
            self.assertEqual(mock_tt_cfg.inputSampleRate, expected_sample_rate)

    async def test_set_sample_rate_multiple_rates(self):
        """Test set_sample_rate with multiple different sample rates."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)

        for sample_rate in [8000, 16000, 24000, 32000, 44100, 48000]:
            turn_analyzer.set_sample_rate(sample_rate)
            self.assertEqual(turn_analyzer.sample_rate, sample_rate)

    async def test_speech_triggered_property(self):
        """Test speech_triggered property."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        self.assertFalse(turn_analyzer.speech_triggered)

        turn_analyzer._speech_triggered = True
        self.assertTrue(turn_analyzer.speech_triggered)

    async def test_params_property(self):
        """Test params property."""
        custom_params = KrispTurnParams(threshold=0.8)
        turn_analyzer = KrispVivaTurn(model_path=self.model_path, params=custom_params)

        self.assertEqual(turn_analyzer.params.threshold, 0.8)
        self.assertIsInstance(turn_analyzer.params, KrispTurnParams)

    async def test_append_audio_without_session(self):
        """Test append_audio returns INCOMPLETE when session is not initialized."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        # Don't set sample rate, so session is None

        audio_data = b"\x00\x01\x02\x03"
        state = turn_analyzer.append_audio(audio_data, is_speech=True)

        self.assertEqual(state, EndOfTurnState.INCOMPLETE)

    async def test_append_audio_with_speech(self):
        """Test append_audio with speech audio."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer.set_sample_rate(16000)

        # Create audio data
        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        audio_data = samples.tobytes()

        state = turn_analyzer.append_audio(audio_data, is_speech=True)

        # Should trigger speech and return INCOMPLETE (probability below threshold)
        self.assertTrue(turn_analyzer._speech_triggered)
        self.assertIsNotNone(turn_analyzer._speech_start_time)
        self.assertIsNone(turn_analyzer._eot_start_time)
        self.assertEqual(state, EndOfTurnState.INCOMPLETE)
        self.mock_tt_session.process.assert_called()

    async def test_append_audio_with_speech_resets_eot_tracking(self):
        """Test that speech resets end-of-turn tracking."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer.set_sample_rate(16000)
        turn_analyzer._speech_triggered = True
        turn_analyzer._eot_start_time = 12345.0  # Some previous time

        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        audio_data = samples.tobytes()

        turn_analyzer.append_audio(audio_data, is_speech=True)

        # EOT start time should be reset
        self.assertIsNone(turn_analyzer._eot_start_time)

    async def test_append_audio_without_speech_starts_eot_tracking(self):
        """Test that non-speech audio starts end-of-turn tracking when speech was triggered."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer.set_sample_rate(16000)
        turn_analyzer._speech_triggered = True

        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        audio_data = samples.tobytes()

        turn_analyzer.append_audio(audio_data, is_speech=False)

        # EOT start time should be set
        self.assertIsNotNone(turn_analyzer._eot_start_time)

    async def test_append_audio_completes_turn_when_probability_above_threshold(self):
        """Test that append_audio returns COMPLETE when probability exceeds threshold."""
        turn_analyzer = KrispVivaTurn(
            model_path=self.model_path, params=KrispTurnParams(threshold=0.5)
        )
        turn_analyzer.set_sample_rate(16000)
        turn_analyzer._speech_triggered = True

        # Mock session to return probability above threshold
        self.mock_tt_session.process.return_value = 0.7

        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        audio_data = samples.tobytes()

        state = turn_analyzer.append_audio(audio_data, is_speech=False)

        self.assertEqual(state, EndOfTurnState.COMPLETE)
        self.assertFalse(turn_analyzer._speech_triggered)  # Should be cleared

    async def test_append_audio_completes_turn_only_when_speech_triggered(self):
        """Test that turn completion only happens when speech was previously triggered."""
        turn_analyzer = KrispVivaTurn(
            model_path=self.model_path, params=KrispTurnParams(threshold=0.5)
        )
        turn_analyzer.set_sample_rate(16000)
        turn_analyzer._speech_triggered = False  # No speech triggered

        # Mock session to return probability above threshold
        self.mock_tt_session.process.return_value = 0.7

        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        audio_data = samples.tobytes()

        state = turn_analyzer.append_audio(audio_data, is_speech=False)

        # Should remain INCOMPLETE because speech was never triggered
        self.assertEqual(state, EndOfTurnState.INCOMPLETE)

    async def test_append_audio_handles_exception(self):
        """Test that append_audio handles exceptions gracefully."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer.set_sample_rate(16000)

        # Make process raise an exception
        self.mock_tt_session.process.side_effect = Exception("Processing error")

        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        audio_data = samples.tobytes()

        state = turn_analyzer.append_audio(audio_data, is_speech=True)

        # Should return INCOMPLETE on error
        self.assertEqual(state, EndOfTurnState.INCOMPLETE)

    async def test_analyze_end_of_turn(self):
        """Test analyze_end_of_turn method."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)

        state, metrics = await turn_analyzer.analyze_end_of_turn()

        self.assertEqual(state, EndOfTurnState.INCOMPLETE)
        self.assertIsNone(metrics)

    async def test_analyze_end_of_turn_with_speech_triggered(self):
        """Test analyze_end_of_turn when speech is triggered."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer._speech_triggered = True

        state, metrics = await turn_analyzer.analyze_end_of_turn()

        self.assertEqual(state, EndOfTurnState.INCOMPLETE)
        self.assertIsNone(metrics)

    async def test_clear(self):
        """Test clear method resets analyzer state."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer._speech_triggered = True
        turn_analyzer._speech_start_time = 12345.0
        turn_analyzer._eot_start_time = 12346.0

        turn_analyzer.clear()

        self.assertFalse(turn_analyzer._speech_triggered)
        self.assertIsNone(turn_analyzer._speech_start_time)
        self.assertIsNone(turn_analyzer._eot_start_time)

    async def test_clear_with_incomplete_state(self):
        """Test _clear method with INCOMPLETE state preserves speech_triggered."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer._speech_triggered = True

        turn_analyzer._clear(EndOfTurnState.INCOMPLETE)

        # Should keep speech_triggered as True for incomplete state
        self.assertTrue(turn_analyzer._speech_triggered)
        self.assertIsNone(turn_analyzer._speech_start_time)
        self.assertIsNone(turn_analyzer._eot_start_time)

    async def test_clear_with_complete_state(self):
        """Test _clear method with COMPLETE state resets speech_triggered."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer._speech_triggered = True

        turn_analyzer._clear(EndOfTurnState.COMPLETE)

        # Should reset speech_triggered for complete state
        self.assertFalse(turn_analyzer._speech_triggered)

    async def test_del_releases_sdk(self):
        """Test that __del__ releases SDK reference."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer.set_sample_rate(16000)

        # Manually call __del__ to test cleanup
        turn_analyzer.__del__()

        # Verify SDK was released
        self.mock_sdk_manager.release.assert_called_once()

    async def test_del_handles_exception(self):
        """Test that __del__ handles exceptions gracefully."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer._sdk_acquired = True

        # Make release raise an exception
        self.mock_sdk_manager.release.side_effect = Exception("Release error")

        # Should not raise exception
        try:
            turn_analyzer.__del__()
        except Exception:
            self.fail("__del__ should handle exceptions gracefully")

    async def test_initialization_with_sample_rate(self):
        """Test initialization with initial sample rate."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path, sample_rate=16000)

        # Sample rate should be set but session not created until set_sample_rate is called
        self.assertEqual(turn_analyzer._init_sample_rate, 16000)
        self.assertEqual(turn_analyzer.sample_rate, 0)  # Not set until set_sample_rate is called

    async def test_multiple_append_audio_calls(self):
        """Test multiple append_audio calls maintain state correctly."""
        turn_analyzer = KrispVivaTurn(
            model_path=self.model_path, params=KrispTurnParams(threshold=0.5)
        )
        turn_analyzer.set_sample_rate(16000)

        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        audio_data = samples.tobytes()

        # First call with speech
        state1 = turn_analyzer.append_audio(audio_data, is_speech=True)
        self.assertEqual(state1, EndOfTurnState.INCOMPLETE)
        self.assertTrue(turn_analyzer._speech_triggered)

        # Second call without speech, probability below threshold
        self.mock_tt_session.process.return_value = 0.3
        state2 = turn_analyzer.append_audio(audio_data, is_speech=False)
        self.assertEqual(state2, EndOfTurnState.INCOMPLETE)
        self.assertTrue(turn_analyzer._speech_triggered)  # Still triggered

        # Third call without speech, probability above threshold
        self.mock_tt_session.process.return_value = 0.7
        state3 = turn_analyzer.append_audio(audio_data, is_speech=False)
        self.assertEqual(state3, EndOfTurnState.COMPLETE)
        self.assertFalse(turn_analyzer._speech_triggered)  # Cleared

    async def test_audio_conversion_to_float32(self):
        """Test that audio is properly converted to float32 format."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer.set_sample_rate(16000)

        # Create enough audio samples for a complete frame (20ms at 16000 Hz = 320 samples)
        samples = np.array([-32768, 0, 32767] * 107, dtype=np.int16)[:320]  # Exactly 320 samples
        audio_data = samples.tobytes()

        turn_analyzer.append_audio(audio_data, is_speech=True)

        # Verify process was called with float32 list
        call_args = self.mock_tt_session.process.call_args
        self.assertIsNotNone(call_args)
        audio_list = call_args[0][0]
        self.assertIsInstance(audio_list, list)
        # Check that values are normalized (should be between -1.0 and 1.0)
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in audio_list))
