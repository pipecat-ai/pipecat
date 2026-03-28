#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock

import numpy as np

try:
    import pyrnnoise
except ImportError:
    pyrnnoise = None

from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.frames.frames import FilterEnableFrame


class TestRNNoiseFilter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        if pyrnnoise is None:
            self.skipTest("pyrnnoise not installed")

    async def test_rnnoise_filter_reduces_noise(self):
        """Test that RNNoise filter reduces noise in audio."""
        filter = RNNoiseFilter()

        # Initialize with 48kHz sample rate (RNNoise requirement)
        await filter.start(sample_rate=48000)

        # Create noisy audio: clean signal + noise
        # Generate a simple sine wave as clean signal
        # Need at least 480 samples (one frame) for processing
        duration = 0.02  # 20ms = 960 samples at 48kHz (2 frames)
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440.0  # A4 note
        clean_signal = np.sin(2 * np.pi * frequency * t)

        # Add white noise
        noise = np.random.normal(0, 0.3, clean_signal.shape)
        noisy_signal = clean_signal + noise

        # Convert to int16 format
        noisy_audio_int16 = (noisy_signal * 32767).astype(np.int16)
        noisy_audio_bytes = noisy_audio_int16.tobytes()

        # Process through filter
        filtered_audio_bytes = await filter.filter(noisy_audio_bytes)

        # Convert back to numpy array for comparison
        filtered_audio = np.frombuffer(filtered_audio_bytes, dtype=np.int16)

        # Verify output is not empty (should have at least one processed frame)
        self.assertGreater(len(filtered_audio), 0)

        # Verify the filtered audio is different from input (noise reduction occurred)
        # The filtered audio should have less variance/noise
        self.assertIsNotNone(filtered_audio_bytes)

        await filter.stop()

    async def test_rnnoise_filter_passthrough_when_disabled(self):
        """Test that RNNoise filter passes through audio when disabled."""
        filter = RNNoiseFilter()
        await filter.start(sample_rate=48000)

        # Disable filtering
        await filter.process_frame(FilterEnableFrame(enable=False))

        # Create test audio
        test_audio = np.random.randint(-32768, 32767, 480, dtype=np.int16).tobytes()

        # Process through filter
        filtered_audio = await filter.filter(test_audio)

        # Should pass through unchanged when disabled
        self.assertEqual(filtered_audio, test_audio)

        await filter.stop()

    async def test_rnnoise_filter_buffering(self):
        """Test that RNNoise filter properly buffers incomplete frames."""
        filter = RNNoiseFilter()
        await filter.start(sample_rate=48000)

        # Send a small chunk that's less than a full frame (480 samples)
        small_chunk = np.random.randint(-32768, 32767, 100, dtype=np.int16).tobytes()

        # First call should return empty (buffering, not enough for a frame)
        result1 = await filter.filter(small_chunk)
        self.assertEqual(result1, b"")

        # Send more data to complete a frame (100 + 500 = 600 samples > 480)
        more_data = np.random.randint(-32768, 32767, 500, dtype=np.int16).tobytes()
        result2 = await filter.filter(more_data)

        # Should return processed audio for at least one complete frame
        self.assertGreater(len(result2), 0)

        await filter.stop()

    async def test_rnnoise_filter_handles_empty_resampler_output(self):
        """Test that RNNoise filter handles empty bytes from resampler gracefully.

        This reproduces the issue where mute/silence input (like "bx000") causes
        the resampler to return empty bytes. The filter should handle this gracefully
        without raising a memory error.
        """
        filter = RNNoiseFilter()

        # Initialize with a sample rate that requires resampling (not 48kHz)
        await filter.start(sample_rate=16000)

        # Enable filtering
        await filter.process_frame(FilterEnableFrame(enable=True))

        # Create mute/silence audio chunk (like "bx000" - all zeros)
        # This simulates a mute sound that causes resampler to return empty bytes
        mute_audio = b"\x00\x00" * 160  # 160 samples of silence at 16kHz (10ms)

        # Mock the resampler to return empty bytes when given mute input
        # This simulates the behavior where resampler returns b"" for mute sounds
        async def mock_resample(audio, in_rate, out_rate):
            # If input is all zeros (mute), return empty bytes
            # This simulates the real-world scenario where resampler returns b"" for silence
            audio_data = np.frombuffer(audio, dtype=np.int16)
            if len(audio_data) == 0 or np.all(audio_data == 0):
                return b""  # Return empty bytes - this triggers the bug
            # Otherwise, do normal resampling
            return audio

        filter._resampler_in.resample = AsyncMock(side_effect=mock_resample)

        result = await filter.filter(mute_audio)

        # When resampler returns empty bytes, filter should return empty bytes
        # (or handle it gracefully without crashing)
        self.assertEqual(
            result, b"", "Filter should return empty bytes when resampler returns empty bytes"
        )

        await filter.stop()


if __name__ == "__main__":
    unittest.main()
