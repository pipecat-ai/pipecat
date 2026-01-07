#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests for KrispVivaTurn in a pipeline context.

These tests verify that KrispVivaTurn works correctly when integrated
with Pipecat's pipeline framework, simulating real-world usage scenarios.
"""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np

# Mock package version check before importing pipecat
# This allows tests to run in development mode without installed package
_version_patcher = patch("importlib.metadata.version", return_value="0.0.0-dev")
_version_patcher.start()

try:
    from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
    from pipecat.audio.turn.krisp_viva_turn import KrispTurnParams, KrispVivaTurn
    from pipecat.frames.frames import (
        EndFrame,
        InputAudioRawFrame,
        StartFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.tests.utils import run_test

    IMPORTS_AVAILABLE = True
except (ImportError, Exception) as e:
    # If import fails due to missing krisp_audio, mark as unavailable
    if "krisp_audio" in str(e) or "Missing module" in str(e):
        KRISP_AVAILABLE = False
        IMPORTS_AVAILABLE = False
        # Set dummy values to prevent NameError, but these won't be used
        # since the test class will be skipped
        KrispVivaTurn = None
        KrispTurnParams = None
        EndOfTurnState = None
        InputAudioRawFrame = None
        StartFrame = None
        EndFrame = None
        FrameDirection = None
        FrameProcessor = None
        run_test = None
    else:
        raise


# Only define MockTransportProcessor if imports are available
# This prevents TypeError when FrameProcessor is None
if IMPORTS_AVAILABLE:

    class MockTransportProcessor(FrameProcessor):
        """Mock transport processor that simulates BaseInputTransport behavior.

        This processor simulates how an input transport would use the turn analyzer:
        - Sets sample rate on StartFrame
        - Calls turn_analyzer.append_audio() on InputAudioRawFrame with VAD state
        - Handles turn completion events
        """

        def __init__(self, turn_analyzer: KrispVivaTurn):
            """Initialize the mock transport processor.

            Args:
                turn_analyzer: The turn analyzer instance to use.
            """
            super().__init__()
            self._turn_analyzer = turn_analyzer
            self._sample_rate = 16000
            self._vad_state = "QUIET"  # Simulate VAD state

        async def process_frame(self, frame, direction: FrameDirection):
            """Process frames and apply turn analyzer as a real transport would."""
            await super().process_frame(frame, direction)

            if isinstance(frame, StartFrame):
                # Set sample rate for turn analyzer
                self._turn_analyzer.set_sample_rate(self._sample_rate)
                await self.push_frame(frame, direction)

            elif isinstance(frame, EndFrame):
                # Clear turn analyzer state
                self._turn_analyzer.clear()
                await self.push_frame(frame, direction)

            elif isinstance(frame, InputAudioRawFrame):
                # Simulate VAD: assume first half is speech, second half is quiet
                # This is a simplified simulation for testing
                is_speech = self._vad_state == "SPEAKING" or self._vad_state == "STARTING"

                # Process audio through turn analyzer
                end_of_turn_state = self._turn_analyzer.append_audio(frame.audio, is_speech)

                # Handle turn completion
                if end_of_turn_state == EndOfTurnState.COMPLETE:
                    # In real transport, this would trigger turn completion handling
                    pass

                await self.push_frame(frame, direction)

            else:
                # Pass through other frames
                await self.push_frame(frame, direction)

        def set_vad_state(self, state: str):
            """Set VAD state for testing."""
            self._vad_state = state
else:
    # Create a dummy class to prevent NameError when test class is skipped
    MockTransportProcessor = None


@unittest.skipIf(not KRISP_AVAILABLE, "krisp_audio module not available (private dependency)")
class TestKrispVivaTurnIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for KrispVivaTurn in pipeline context."""

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

    async def test_turn_analyzer_in_pipeline_with_speech(self):
        """Test turn analyzer integrated in a pipeline processing speech audio."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        transport = MockTransportProcessor(turn_analyzer)
        transport.set_vad_state("SPEAKING")

        # Create audio data
        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        input_audio = samples.tobytes()

        frames_to_send = [
            StartFrame(),
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
        ]

        expected_down_frames = [
            InputAudioRawFrame,  # Audio frame passed through
        ]

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify turn analyzer processed the audio
        self.mock_tt_session.process.assert_called()
        # Speech should be triggered
        self.assertTrue(turn_analyzer.speech_triggered)

    async def test_turn_analyzer_lifecycle_in_pipeline(self):
        """Test turn analyzer lifecycle (set_sample_rate/clear) in pipeline context."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        transport = MockTransportProcessor(turn_analyzer)

        frames_to_send = [
            StartFrame(),
        ]

        expected_down_frames = []  # No data frames, just lifecycle

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify turn analyzer was initialized with sample rate
        self.assertEqual(turn_analyzer.sample_rate, 16000)
        # Verify session was created
        self.assertIsNotNone(turn_analyzer._tt_session)

    async def test_turn_analyzer_detects_turn_completion(self):
        """Test turn analyzer detects turn completion when probability exceeds threshold."""
        turn_analyzer = KrispVivaTurn(
            model_path=self.model_path, params=KrispTurnParams(threshold=0.5)
        )
        transport = MockTransportProcessor(turn_analyzer)

        # Start with speech
        transport.set_vad_state("SPEAKING")

        # Create audio data
        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        input_audio = samples.tobytes()

        # First frame: speech (triggers speech detection)
        frames_to_send = [
            StartFrame(),
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
        ]

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=[InputAudioRawFrame],
        )

        # Verify speech was triggered
        self.assertTrue(turn_analyzer.speech_triggered)

        # Second frame: quiet with high probability (should complete turn)
        transport.set_vad_state("QUIET")
        self.mock_tt_session.process.return_value = 0.7  # Above threshold

        frames_to_send = [
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
        ]

        # Manually process to check turn completion
        end_of_turn_state = turn_analyzer.append_audio(input_audio, is_speech=False)
        self.assertEqual(end_of_turn_state, EndOfTurnState.COMPLETE)
        self.assertFalse(turn_analyzer.speech_triggered)  # Should be cleared

    async def test_turn_analyzer_with_streaming_audio(self):
        """Test turn analyzer with realistic streaming audio scenario."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        transport = MockTransportProcessor(turn_analyzer)
        transport.set_vad_state("SPEAKING")

        # Simulate streaming audio: multiple small chunks
        audio_chunks = []
        for i in range(5):
            samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
            audio_chunks.append(samples.tobytes())

        frames_to_send = [StartFrame()]
        for chunk in audio_chunks:
            frames_to_send.append(
                InputAudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1)
            )

        expected_down_frames = [InputAudioRawFrame] * 5

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify all chunks were processed
        self.assertEqual(self.mock_tt_session.process.call_count, 5)
        self.assertTrue(turn_analyzer.speech_triggered)

    async def test_turn_analyzer_speech_to_silence_transition(self):
        """Test turn analyzer handles speech to silence transition correctly."""
        turn_analyzer = KrispVivaTurn(
            model_path=self.model_path, params=KrispTurnParams(threshold=0.5)
        )
        transport = MockTransportProcessor(turn_analyzer)

        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        input_audio = samples.tobytes()

        # Start with speech
        transport.set_vad_state("SPEAKING")
        frames_to_send = [
            StartFrame(),
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
        ]

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=[InputAudioRawFrame],
        )

        # Verify speech was triggered
        self.assertTrue(turn_analyzer.speech_triggered)
        self.assertIsNotNone(turn_analyzer._speech_start_time)

        # Transition to silence
        transport.set_vad_state("QUIET")
        # Keep probability below threshold
        self.mock_tt_session.process.return_value = 0.3

        end_of_turn_state = turn_analyzer.append_audio(input_audio, is_speech=False)

        # Should still be incomplete (probability below threshold)
        self.assertEqual(end_of_turn_state, EndOfTurnState.INCOMPLETE)
        self.assertTrue(turn_analyzer.speech_triggered)  # Still triggered
        self.assertIsNotNone(turn_analyzer._eot_start_time)  # EOT tracking started

    async def test_turn_analyzer_error_recovery_in_pipeline(self):
        """Test turn analyzer error handling in pipeline context."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        transport = MockTransportProcessor(turn_analyzer)
        transport.set_vad_state("SPEAKING")

        # Make session.process raise an exception
        self.mock_tt_session.process.side_effect = Exception("Processing error")

        samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
        input_audio = samples.tobytes()

        frames_to_send = [
            StartFrame(),
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
        ]

        expected_down_frames = [
            InputAudioRawFrame,  # Should still pass through
        ]

        # Should not raise exception, turn analyzer should handle error gracefully
        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Should return INCOMPLETE on error
        end_of_turn_state = turn_analyzer.append_audio(input_audio, is_speech=True)
        self.assertEqual(end_of_turn_state, EndOfTurnState.INCOMPLETE)

    async def test_turn_analyzer_multiple_sample_rates_in_pipeline(self):
        """Test turn analyzer with different sample rates in pipeline."""
        for sample_rate in [16000, 24000, 48000]:
            turn_analyzer = KrispVivaTurn(model_path=self.model_path)
            transport = MockTransportProcessor(turn_analyzer)
            transport._sample_rate = sample_rate
            transport.set_vad_state("SPEAKING")

            samples = np.random.randint(-32768, 32767, size=sample_rate // 50, dtype=np.int16)
            input_audio = samples.tobytes()

            frames_to_send = [
                StartFrame(),
                InputAudioRawFrame(audio=input_audio, sample_rate=sample_rate, num_channels=1),
            ]

            expected_down_frames = [
                InputAudioRawFrame,
            ]

            await run_test(
                transport,
                frames_to_send=frames_to_send,
                expected_down_frames=expected_down_frames,
            )

            # Verify sample rate was set correctly
            self.assertEqual(turn_analyzer.sample_rate, sample_rate)

            # Reset mock for next iteration
            self.mock_tt_session.process.reset_mock()

    async def test_turn_analyzer_with_different_thresholds(self):
        """Test turn analyzer with different probability thresholds."""
        for threshold in [0.3, 0.5, 0.7]:
            turn_analyzer = KrispVivaTurn(
                model_path=self.model_path, params=KrispTurnParams(threshold=threshold)
            )
            turn_analyzer.set_sample_rate(16000)
            turn_analyzer._speech_triggered = True

            samples = np.random.randint(-32768, 32767, size=320, dtype=np.int16)
            input_audio = samples.tobytes()

            # Set probability just above threshold
            test_probability = threshold + 0.1
            self.mock_tt_session.process.return_value = test_probability

            end_of_turn_state = turn_analyzer.append_audio(input_audio, is_speech=False)

            # Should complete turn
            self.assertEqual(end_of_turn_state, EndOfTurnState.COMPLETE)

            # Reset for next iteration
            turn_analyzer.clear()
            self.mock_tt_session.process.reset_mock()

    async def test_turn_analyzer_clear_resets_state(self):
        """Test that clear() properly resets turn analyzer state."""
        turn_analyzer = KrispVivaTurn(model_path=self.model_path)
        turn_analyzer.set_sample_rate(16000)
        turn_analyzer._speech_triggered = True
        turn_analyzer._speech_start_time = 12345.0
        turn_analyzer._eot_start_time = 12346.0

        turn_analyzer.clear()

        self.assertFalse(turn_analyzer.speech_triggered)
        self.assertIsNone(turn_analyzer._speech_start_time)
        self.assertIsNone(turn_analyzer._eot_start_time)

    async def test_turn_analyzer_with_different_frame_durations(self):
        """Test turn analyzer with different frame duration configurations."""
        for frame_duration_ms in [10, 20, 30]:
            # Reset mock config for each iteration
            mock_tt_cfg = MagicMock()
            self.mock_krisp_audio.TtSessionConfig.return_value = mock_tt_cfg

            turn_analyzer = KrispVivaTurn(
                model_path=self.model_path,
                params=KrispTurnParams(frame_duration_ms=frame_duration_ms),
            )
            turn_analyzer.set_sample_rate(16000)

            # Verify correct frame duration was set
            expected_enum = {
                10: "10ms",
                20: "20ms",
                30: "30ms",
            }[frame_duration_ms]
            self.assertEqual(mock_tt_cfg.inputFrameDuration, expected_enum)

            # Clean up
            turn_analyzer.__del__()


if __name__ == "__main__":
    unittest.main()
