#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

# pyright: reportConstantRedefinition=false
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalCall=false

import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipecat.transports.smallwebrtc.transport import RawAudioTrack

try:
    from pipecat.transports.smallwebrtc.transport import RawAudioTrack

    WEBRTC_AVAILABLE = True
except (ImportError, Exception):
    WEBRTC_AVAILABLE = False
    RawAudioTrack = None  # type: ignore[misc,assignment]


@unittest.skipUnless(WEBRTC_AVAILABLE, "webrtc dependencies not installed")
class TestRawAudioTrack(unittest.IsolatedAsyncioTestCase):
    """Tests for the RawAudioTrack class."""

    def test_default_chunk_size_is_10ms(self):
        """Test that default chunk size is 10ms (num_10ms_chunks=1)."""
        sample_rate = 16000
        track = RawAudioTrack(sample_rate=sample_rate)

        # 10ms at 16kHz = 160 samples, 2 bytes per sample = 320 bytes
        expected_bytes = int(sample_rate * 10 / 1000) * 2
        self.assertEqual(track._bytes_per_chunk, expected_bytes)
        self.assertEqual(track._bytes_per_chunk, 320)

    def test_custom_chunk_size_40ms(self):
        """Test that num_10ms_chunks=4 produces 40ms chunks."""
        sample_rate = 16000
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=4)

        # 40ms at 16kHz = 640 samples, 2 bytes per sample = 1280 bytes
        expected_bytes = int(sample_rate * 40 / 1000) * 2
        self.assertEqual(track._bytes_per_chunk, expected_bytes)
        self.assertEqual(track._bytes_per_chunk, 1280)

    def test_custom_chunk_size_20ms(self):
        """Test that num_10ms_chunks=2 produces 20ms chunks."""
        sample_rate = 16000
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=2)

        # 20ms at 16kHz = 320 samples, 2 bytes per sample = 640 bytes
        expected_bytes = int(sample_rate * 20 / 1000) * 2
        self.assertEqual(track._bytes_per_chunk, expected_bytes)
        self.assertEqual(track._bytes_per_chunk, 640)

    async def test_add_audio_bytes_queues_correct_chunks(self):
        """Test that add_audio_bytes breaks audio into correct chunk sizes."""
        sample_rate = 16000
        num_chunks = 4  # 40ms
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=num_chunks)

        # Create 80ms of audio (2 chunks of 40ms each)
        audio_bytes = bytes(track._bytes_per_chunk * 2)
        track.add_audio_bytes(audio_bytes)

        # Should have exactly 2 chunks in the queue
        self.assertEqual(len(track._chunk_queue), 2)

        # Each chunk should be the correct size
        chunk1, _ = track._chunk_queue[0]
        chunk2, _ = track._chunk_queue[1]
        self.assertEqual(len(chunk1), track._bytes_per_chunk)
        self.assertEqual(len(chunk2), track._bytes_per_chunk)

    async def test_add_audio_bytes_rejects_invalid_size(self):
        """Test that add_audio_bytes rejects audio not a multiple of chunk size."""
        sample_rate = 16000
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=4)

        # Create audio that's not a multiple of 40ms chunk size
        invalid_audio = bytes(track._bytes_per_chunk + 100)

        with self.assertRaises(ValueError) as ctx:
            track.add_audio_bytes(invalid_audio)

        self.assertIn("40ms", str(ctx.exception))

    async def test_recv_returns_correct_frame_size(self):
        """Test that recv() returns AudioFrames with correct sample count."""
        sample_rate = 16000
        num_chunks = 4  # 40ms
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=num_chunks)

        # Add one 40ms chunk of audio
        audio_bytes = bytes(track._bytes_per_chunk)
        track.add_audio_bytes(audio_bytes)

        # Receive the frame
        frame = await track.recv()

        # Frame should have correct number of samples (40ms worth)
        expected_samples = int(sample_rate * 40 / 1000)  # 640 samples
        self.assertEqual(frame.samples, expected_samples)

    async def test_recv_silence_has_correct_size(self):
        """Test that silence frames have correct size when queue is empty."""
        sample_rate = 16000
        num_chunks = 4  # 40ms
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=num_chunks)

        # Don't add any audio - should get silence
        frame = await track.recv()

        # Silence frame should have correct number of samples
        expected_samples = int(sample_rate * 40 / 1000)  # 640 samples
        self.assertEqual(frame.samples, expected_samples)

    async def test_timestamp_advances_by_chunk_samples(self):
        """Test that timestamp advances correctly based on chunk size."""
        sample_rate = 16000
        num_chunks = 4  # 40ms
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=num_chunks)

        # Receive first frame and check its timestamp
        frame1 = await track.recv()
        # Receive second frame
        frame2 = await track.recv()

        # Timestamp should advance by samples_per_chunk between frames
        self.assertIsNotNone(frame1.pts)
        self.assertIsNotNone(frame2.pts)
        expected_samples = int(sample_rate * 40 / 1000)  # 640 samples
        self.assertEqual(frame2.pts - frame1.pts, expected_samples)

    def test_different_sample_rates(self):
        """Test chunk size calculation at different sample rates."""
        test_cases = [
            (8000, 4, 640),  # 8kHz, 40ms = 320 samples * 2 bytes = 640 bytes
            (16000, 4, 1280),  # 16kHz, 40ms = 640 samples * 2 bytes = 1280 bytes
            (24000, 4, 1920),  # 24kHz, 40ms = 960 samples * 2 bytes = 1920 bytes
            (48000, 4, 3840),  # 48kHz, 40ms = 1920 samples * 2 bytes = 3840 bytes
        ]

        for sample_rate, num_chunks, expected_bytes in test_cases:
            track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=num_chunks)
            self.assertEqual(track._bytes_per_chunk, expected_bytes)

    def test_invalid_num_10ms_chunks_zero(self):
        """Test that num_10ms_chunks=0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RawAudioTrack(sample_rate=16000, num_10ms_chunks=0)

        self.assertIn("positive integer", str(ctx.exception))

    def test_invalid_num_10ms_chunks_negative(self):
        """Test that negative num_10ms_chunks raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RawAudioTrack(sample_rate=16000, num_10ms_chunks=-1)

        self.assertIn("positive integer", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
