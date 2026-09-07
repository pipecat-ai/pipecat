#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import math
import unittest
from unittest.mock import patch

import numpy as np

from pipecat.audio.volume import VOLUME_WINDOW_SECS, AudioVolumeTracker

SAMPLE_RATE = 16000
CHUNK_NUM_BYTES = 320 * 2  # 20ms


def audio(num_samples: int, amplitude: int, seed: int = 0) -> bytes:
    """Generate mono 16-bit noise of the given amplitude."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, amplitude, num_samples).astype(np.int16).tobytes()


def window_num_bytes(sample_rate: int = SAMPLE_RATE) -> int:
    return math.ceil(VOLUME_WINDOW_SECS * sample_rate) * 2


def feed(tracker: AudioVolumeTracker, data: bytes, sample_rate: int = SAMPLE_RATE) -> list[float]:
    """Feed audio in 20ms chunks and return the volume after each one."""
    volumes = []
    for i in range(0, len(data), CHUNK_NUM_BYTES):
        tracker.update(data[i : i + CHUNK_NUM_BYTES], sample_rate)
        volumes.append(tracker.volume)
    return volumes


class TestAudioVolumeTracker(unittest.IsolatedAsyncioTestCase):
    async def test_volume_is_zero_until_window_fills(self):
        tracker = AudioVolumeTracker()
        window_samples = math.ceil(VOLUME_WINDOW_SECS * SAMPLE_RATE)

        # One chunk short of a full window.
        volumes = feed(tracker, audio(window_samples - 320, amplitude=8000))
        self.assertTrue(all(v == 0.0 for v in volumes))
        self.assertEqual(tracker.volume, 0.0)

        # The chunk that completes the window produces a measurement.
        tracker.update(audio(320, amplitude=8000), SAMPLE_RATE)
        self.assertGreater(tracker.volume, 0.0)

    async def test_loud_audio_reads_louder_than_quiet_audio(self):
        loud = AudioVolumeTracker()
        quiet = AudioVolumeTracker()
        num_samples = SAMPLE_RATE  # 1s, comfortably longer than the window

        feed(loud, audio(num_samples, amplitude=8000))
        feed(quiet, audio(num_samples, amplitude=50))

        self.assertGreater(loud.volume, quiet.volume)
        self.assertGreaterEqual(loud.volume, 0.0)
        self.assertLessEqual(loud.volume, 1.0)

    async def test_silence_reads_zero(self):
        tracker = AudioVolumeTracker()
        feed(tracker, b"\x00\x00" * SAMPLE_RATE)
        self.assertEqual(tracker.volume, 0.0)

    async def test_window_is_bounded(self):
        tracker = AudioVolumeTracker()
        feed(tracker, audio(SAMPLE_RATE * 5, amplitude=8000))
        self.assertEqual(len(tracker._buffer), window_num_bytes())

    async def test_chunk_larger_than_window(self):
        tracker = AudioVolumeTracker()
        tracker.update(audio(SAMPLE_RATE * 3, amplitude=8000), SAMPLE_RATE)
        self.assertGreater(tracker.volume, 0.0)
        self.assertEqual(len(tracker._buffer), window_num_bytes())

    async def test_volume_is_measured_once_per_update(self):
        tracker = AudioVolumeTracker()
        tracker.update(audio(SAMPLE_RATE, amplitude=8000), SAMPLE_RATE)

        with patch("pipecat.audio.volume.calculate_audio_volume", return_value=0.5) as measure:
            self.assertEqual(tracker.volume, 0.5)
            self.assertEqual(tracker.volume, 0.5)
            self.assertEqual(measure.call_count, 1)

            # Fresh audio invalidates the cached measurement.
            tracker.update(audio(320, amplitude=8000), SAMPLE_RATE)
            self.assertEqual(tracker.volume, 0.5)
            self.assertEqual(measure.call_count, 2)

    async def test_reading_less_often_than_updating_measures_latest_window(self):
        periodic = AudioVolumeTracker()
        every_chunk = AudioVolumeTracker()
        data = audio(SAMPLE_RATE, amplitude=8000)

        for i in range(0, len(data), CHUNK_NUM_BYTES):
            periodic.update(data[i : i + CHUNK_NUM_BYTES], SAMPLE_RATE)
        self.assertEqual(periodic.volume, feed(every_chunk, data)[-1])

    async def test_sample_rate_change_discards_window(self):
        tracker = AudioVolumeTracker()
        feed(tracker, audio(SAMPLE_RATE, amplitude=8000))
        self.assertGreater(tracker.volume, 0.0)

        tracker.update(audio(320, amplitude=8000), 8000)
        self.assertEqual(tracker.volume, 0.0)
        self.assertEqual(len(tracker._buffer), 320 * 2)

    async def test_reset(self):
        tracker = AudioVolumeTracker()
        feed(tracker, audio(SAMPLE_RATE, amplitude=8000))
        self.assertGreater(tracker.volume, 0.0)

        tracker.reset()
        self.assertEqual(tracker.volume, 0.0)
        self.assertEqual(feed(tracker, audio(320, amplitude=8000))[0], 0.0)

    async def test_supported_sample_rates(self):
        for sample_rate in (8000, 16000, 22050, 24000, 44100, 48000):
            with self.subTest(sample_rate=sample_rate):
                tracker = AudioVolumeTracker()
                feed(tracker, audio(sample_rate, amplitude=8000), sample_rate)
                self.assertGreater(tracker.volume, 0.0)

    async def test_window_is_never_short_of_a_gating_block(self):
        # Rates where 400ms isn't a whole number of samples: rounding the window
        # down would leave it just short of a gating block, which is rejected.
        for sample_rate in (7999, 11999, 16001, 44101):
            with self.subTest(sample_rate=sample_rate):
                tracker = AudioVolumeTracker()
                feed(tracker, audio(sample_rate, amplitude=8000), sample_rate)
                self.assertGreaterEqual(len(tracker._buffer) / 2 / sample_rate, VOLUME_WINDOW_SECS)
                self.assertGreater(tracker.volume, 0.0)


if __name__ == "__main__":
    unittest.main()
