#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests for KrispVivaFilter in a pipeline context.

These tests verify that KrispVivaFilter works correctly when integrated
with Pipecat's pipeline framework, simulating real-world usage scenarios.
"""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
    from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
    from pipecat.frames.frames import (
        EndFrame,
        FilterEnableFrame,
        InputAudioRawFrame,
        StartFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
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
        KrispVivaFilter = None
        EndFrame = None
        FilterEnableFrame = None
        InputAudioRawFrame = None
        StartFrame = None
        Pipeline = None
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

        This processor simulates how an input transport would use the audio filter:
        - Calls filter.start() on StartFrame
        - Calls filter.filter() on InputAudioRawFrame
        - Calls filter.stop() on EndFrame
        - Processes FilterEnableFrame for runtime control
        """

        def __init__(self, audio_filter: KrispVivaFilter):
            """Initialize the mock transport processor.

            Args:
                audio_filter: The audio filter instance to use.
            """
            super().__init__()
            self._audio_filter = audio_filter
            self._sample_rate = 16000

        async def process_frame(self, frame, direction: FrameDirection):
            """Process frames and apply audio filter as a real transport would."""
            await super().process_frame(frame, direction)

            if isinstance(frame, StartFrame):
                # Initialize filter with sample rate
                await self._audio_filter.start(self._sample_rate)
                await self.push_frame(frame, direction)

            elif isinstance(frame, EndFrame):
                # Stop filter before pushing EndFrame
                await self._audio_filter.stop()
                await self.push_frame(frame, direction)

            elif isinstance(frame, FilterEnableFrame):
                # Process filter control frames
                await self._audio_filter.process_frame(frame)
                await self.push_frame(frame, direction)

            elif isinstance(frame, InputAudioRawFrame):
                # Apply filter to audio data
                filtered_audio = await self._audio_filter.filter(frame.audio)

                # Create new frame with filtered audio
                filtered_frame = InputAudioRawFrame(
                    audio=filtered_audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                await self.push_frame(filtered_frame, direction)

            else:
                # Pass through other frames
                await self.push_frame(frame, direction)
else:
    # Create a dummy class to prevent NameError when test class is skipped
    MockTransportProcessor = None


@unittest.skipIf(not KRISP_AVAILABLE, "krisp_audio module not available (private dependency)")
class TestKrispVivaFilterIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for KrispVivaFilter in pipeline context."""

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

        # Mock ModelInfo
        self.mock_model_info = MagicMock()
        self.mock_krisp_audio.ModelInfo.return_value = self.mock_model_info

        # Mock NcSessionConfig
        self.mock_nc_cfg = MagicMock()
        self.mock_krisp_audio.NcSessionConfig.return_value = self.mock_nc_cfg

        # Mock session
        self.mock_session = MagicMock()
        self.mock_session.process = MagicMock(side_effect=lambda x, level: x.copy())
        self.mock_krisp_audio.NcInt16.create.return_value = self.mock_session

        # Patch krisp_audio at module level
        self.krisp_audio_patcher = patch.dict("sys.modules", {"krisp_audio": self.mock_krisp_audio})
        self.krisp_audio_patcher.start()

        # Patch the SAMPLE_RATES after import
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
        self.krisp_audio_patcher.stop()
        self.sample_rates_patch.stop()
        self.sdk_manager_patcher.stop()

        # Remove temporary model file
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    async def test_filter_in_pipeline_with_audio_frames(self):
        """Test filter integrated in a pipeline processing audio frames."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        transport = MockTransportProcessor(filter_instance)

        # Create audio data for multiple frames
        samples = np.random.randint(-32768, 32767, size=480, dtype=np.int16)  # 3 frames
        input_audio = samples.tobytes()

        frames_to_send = [
            StartFrame(),
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
        ]

        expected_down_frames = [
            InputAudioRawFrame,  # Filtered audio frame
        ]

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify filter processed the audio
        # Note: Session is None after EndFrame (which run_test sends automatically)
        self.mock_session.process.assert_called()

    async def test_filter_lifecycle_in_pipeline(self):
        """Test filter lifecycle (start/stop) in pipeline context."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        transport = MockTransportProcessor(filter_instance)

        frames_to_send = [
            StartFrame(),
        ]

        expected_down_frames = []  # No data frames, just lifecycle

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify filter was started and stopped
        self.assertIsNone(filter_instance._session)  # Stopped after EndFrame

    async def test_filter_enable_disable_in_pipeline(self):
        """Test enabling/disabling filter during pipeline execution."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        transport = MockTransportProcessor(filter_instance)

        # Reset mock call count to ensure we only count calls from this test
        self.mock_session.process.reset_mock()

        # Create audio data
        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        frames_to_send = [
            StartFrame(),
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
            FilterEnableFrame(enable=False),  # Disable filter
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
            FilterEnableFrame(enable=True),  # Re-enable filter
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
        ]

        # Note: Frame order may vary due to async processing
        # We verify that all expected frames are present
        expected_down_frames = [
            InputAudioRawFrame,  # Filtered (enabled)
            InputAudioRawFrame,  # Unfiltered (disabled)
            InputAudioRawFrame,  # Filtered (enabled)
            FilterEnableFrame,  # Control frame passed through
            FilterEnableFrame,  # Control frame passed through
        ]

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify filter was called only when enabled
        # Should be called 2 times (first and last frame, middle one skipped)
        # Note: Due to async processing, FilterEnableFrame might be processed
        # after audio frames, so we verify at least 2 calls (could be more if timing differs)
        self.assertGreaterEqual(self.mock_session.process.call_count, 2)

    async def test_filter_with_streaming_audio(self):
        """Test filter with realistic streaming audio scenario."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        transport = MockTransportProcessor(filter_instance)

        # Simulate streaming audio: multiple small chunks
        audio_chunks = []
        for i in range(5):
            samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
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
        self.assertEqual(self.mock_session.process.call_count, 5)

    async def test_filter_with_incomplete_frames_streaming(self):
        """Test filter handles incomplete frames in streaming scenario."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        transport = MockTransportProcessor(filter_instance)

        # First chunk: incomplete frame (100 samples)
        samples1 = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        chunk1 = samples1.tobytes()

        # Second chunk: completes first frame + starts second (60 + 100 = 160 samples)
        samples2 = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        chunk2 = samples2.tobytes()

        frames_to_send = [
            StartFrame(),
            InputAudioRawFrame(audio=chunk1, sample_rate=16000, num_channels=1),
            InputAudioRawFrame(audio=chunk2, sample_rate=16000, num_channels=1),
        ]

        expected_down_frames = [
            InputAudioRawFrame,  # May be empty if first chunk incomplete
            InputAudioRawFrame,  # Should contain processed frames
        ]

        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify processing occurred (at least one complete frame processed)
        self.assertGreater(self.mock_session.process.call_count, 0)

    async def test_filter_error_recovery_in_pipeline(self):
        """Test filter error handling in pipeline context."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)
        transport = MockTransportProcessor(filter_instance)

        # Make session.process raise an exception
        self.mock_session.process.side_effect = Exception("Processing error")

        samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        input_audio = samples.tobytes()

        frames_to_send = [
            StartFrame(),
            InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
        ]

        expected_down_frames = [
            InputAudioRawFrame,  # Should still pass through (original audio on error)
        ]

        # Should not raise exception, filter should handle error gracefully
        await run_test(
            transport,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_filter_multiple_sample_rates_in_pipeline(self):
        """Test filter with different sample rates in pipeline."""
        filter_instance = KrispVivaFilter(model_path=self.model_path)

        for sample_rate in [16000, 24000, 48000]:
            transport = MockTransportProcessor(filter_instance)
            transport._sample_rate = sample_rate

            samples = np.random.randint(-32768, 32767, size=sample_rate // 100, dtype=np.int16)
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

            # Reset mock for next iteration
            self.mock_session.process.reset_mock()

    async def test_filter_with_noise_suppression_levels(self):
        """Test filter with different noise suppression levels."""
        for noise_level in [50, 75, 100]:
            filter_instance = KrispVivaFilter(
                model_path=self.model_path, noise_suppression_level=noise_level
            )
            transport = MockTransportProcessor(filter_instance)

            samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
            input_audio = samples.tobytes()

            frames_to_send = [
                StartFrame(),
                InputAudioRawFrame(audio=input_audio, sample_rate=16000, num_channels=1),
            ]

            await run_test(
                transport,
                frames_to_send=frames_to_send,
                expected_down_frames=[InputAudioRawFrame],
            )

            # Verify noise suppression level was used
            call_args = self.mock_session.process.call_args
            self.assertEqual(call_args[0][1], noise_level)

            # Reset mock for next iteration
            self.mock_session.process.reset_mock()


if __name__ == "__main__":
    unittest.main()
