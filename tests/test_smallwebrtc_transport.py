#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.transports.smallwebrtc.transport import RawAudioTrack


class TestRawAudioTrack:
    """Tests for the RawAudioTrack class."""

    def test_default_chunk_size_is_10ms(self):
        """Test that default chunk size is 10ms (num_10ms_chunks=1)."""
        sample_rate = 16000
        track = RawAudioTrack(sample_rate=sample_rate)

        # 10ms at 16kHz = 160 samples, 2 bytes per sample = 320 bytes
        expected_bytes = int(sample_rate * 10 / 1000) * 2
        assert track._bytes_per_chunk == expected_bytes
        assert track._bytes_per_chunk == 320

    def test_custom_chunk_size_40ms(self):
        """Test that num_10ms_chunks=4 produces 40ms chunks."""
        sample_rate = 16000
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=4)

        # 40ms at 16kHz = 640 samples, 2 bytes per sample = 1280 bytes
        expected_bytes = int(sample_rate * 40 / 1000) * 2
        assert track._bytes_per_chunk == expected_bytes
        assert track._bytes_per_chunk == 1280

    def test_custom_chunk_size_20ms(self):
        """Test that num_10ms_chunks=2 produces 20ms chunks."""
        sample_rate = 16000
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=2)

        # 20ms at 16kHz = 320 samples, 2 bytes per sample = 640 bytes
        expected_bytes = int(sample_rate * 20 / 1000) * 2
        assert track._bytes_per_chunk == expected_bytes
        assert track._bytes_per_chunk == 640

    @pytest.mark.asyncio
    async def test_add_audio_bytes_queues_correct_chunks(self):
        """Test that add_audio_bytes breaks audio into correct chunk sizes."""
        sample_rate = 16000
        num_chunks = 4  # 40ms
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=num_chunks)

        # Create 80ms of audio (2 chunks of 40ms each)
        audio_bytes = bytes(track._bytes_per_chunk * 2)
        track.add_audio_bytes(audio_bytes)

        # Should have exactly 2 chunks in the queue
        assert len(track._chunk_queue) == 2

        # Each chunk should be the correct size
        chunk1, _ = track._chunk_queue[0]
        chunk2, _ = track._chunk_queue[1]
        assert len(chunk1) == track._bytes_per_chunk
        assert len(chunk2) == track._bytes_per_chunk

    @pytest.mark.asyncio
    async def test_add_audio_bytes_rejects_invalid_size(self):
        """Test that add_audio_bytes rejects audio not a multiple of chunk size."""
        sample_rate = 16000
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=4)

        # Create audio that's not a multiple of 40ms chunk size
        invalid_audio = bytes(track._bytes_per_chunk + 100)

        with pytest.raises(ValueError) as exc_info:
            track.add_audio_bytes(invalid_audio)

        assert "40ms" in str(exc_info.value)

    @pytest.mark.asyncio
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
        assert frame.samples == expected_samples

    @pytest.mark.asyncio
    async def test_recv_silence_has_correct_size(self):
        """Test that silence frames have correct size when queue is empty."""
        sample_rate = 16000
        num_chunks = 4  # 40ms
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=num_chunks)

        # Don't add any audio - should get silence
        frame = await track.recv()

        # Silence frame should have correct number of samples
        expected_samples = int(sample_rate * 40 / 1000)  # 640 samples
        assert frame.samples == expected_samples

    @pytest.mark.asyncio
    async def test_timestamp_advances_by_chunk_samples(self):
        """Test that timestamp advances correctly based on chunk size."""
        sample_rate = 16000
        num_chunks = 4  # 40ms
        track = RawAudioTrack(sample_rate=sample_rate, num_10ms_chunks=num_chunks)

        # Initial timestamp should be 0
        assert track._timestamp == 0

        # Receive one frame (silence is fine)
        await track.recv()

        # Timestamp should advance by samples_per_chunk
        expected_samples = int(sample_rate * 40 / 1000)  # 640 samples
        assert track._timestamp == expected_samples

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
            assert track._bytes_per_chunk == expected_bytes
