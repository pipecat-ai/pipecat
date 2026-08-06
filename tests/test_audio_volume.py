#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import numpy as np

from pipecat.audio.volume import VOLUME_WINDOW_SECS, AudioVolumeTracker

SAMPLE_RATE = 16000
CHUNK_NUM_BYTES = 320 * 2  # 20ms


def audio(num_samples: int, amplitude: int, seed: int = 0) -> bytes:
    """Generate mono 16-bit noise of the given amplitude."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, amplitude, num_samples).astype(np.int16).tobytes()


def feed(tracker: AudioVolumeTracker, data: bytes, sample_rate: int = SAMPLE_RATE) -> list[float]:
    """Feed audio in 20ms chunks and return the volume after each one."""
    return [
        tracker.update(data[i : i + CHUNK_NUM_BYTES], sample_rate)
        for i in range(0, len(data), CHUNK_NUM_BYTES)
    ]


class TestAudioVolumeTracker(unittest.IsolatedAsyncioTestCase):
    async def test_volume_is_zero_until_window_fills(self):
        tracker = AudioVolumeTracker()
        window_samples = round(VOLUME_WINDOW_SECS * SAMPLE_RATE)

        # One chunk short of a full window.
        volumes = feed(tracker, audio(window_samples - 320, amplitude=8000))
        self.assertTrue(all(v == 0.0 for v in volumes))
        self.assertEqual(tracker.volume, 0.0)

        # The chunk that completes the window produces a measurement.
        self.assertGreater(tracker.update(audio(320, amplitude=8000), SAMPLE_RATE), 0.0)

    async def test_loud_audio_reads_louder_than_quiet_audio(self):
        loud = AudioVolumeTracker(smoothing_factor=1.0)
        quiet = AudioVolumeTracker(smoothing_factor=1.0)
        num_samples = SAMPLE_RATE  # 1s, comfortably longer than the window

        feed(loud, audio(num_samples, amplitude=8000))
        feed(quiet, audio(num_samples, amplitude=50))

        self.assertGreater(loud.volume, quiet.volume)
        self.assertGreaterEqual(loud.volume, 0.0)
        self.assertLessEqual(loud.volume, 1.0)

    async def test_silence_reads_zero(self):
        tracker = AudioVolumeTracker(smoothing_factor=1.0)
        feed(tracker, b"\x00\x00" * SAMPLE_RATE)
        self.assertEqual(tracker.volume, 0.0)

    async def test_window_is_bounded(self):
        tracker = AudioVolumeTracker()
        feed(tracker, audio(SAMPLE_RATE * 5, amplitude=8000))
        self.assertEqual(len(tracker._buffer), round(VOLUME_WINDOW_SECS * SAMPLE_RATE) * 2)

    async def test_chunk_larger_than_window(self):
        tracker = AudioVolumeTracker(smoothing_factor=1.0)
        volume = tracker.update(audio(SAMPLE_RATE * 3, amplitude=8000), SAMPLE_RATE)
        self.assertGreater(volume, 0.0)
        self.assertEqual(len(tracker._buffer), round(VOLUME_WINDOW_SECS * SAMPLE_RATE) * 2)

    async def test_smoothing_converges_towards_measured_volume(self):
        smoothed = AudioVolumeTracker(smoothing_factor=0.2)
        unsmoothed = AudioVolumeTracker(smoothing_factor=1.0)
        data = audio(SAMPLE_RATE, amplitude=8000)

        # The first measurement is pulled towards the initial volume of 0, so a
        # smoothed tracker lags behind an unsmoothed one.
        first_smoothed = [v for v in feed(smoothed, data) if v > 0.0][0]
        first_unsmoothed = [v for v in feed(unsmoothed, data) if v > 0.0][0]
        self.assertLess(first_smoothed, first_unsmoothed)

        # Given enough audio it catches up.
        feed(smoothed, audio(SAMPLE_RATE * 3, amplitude=8000, seed=1))
        self.assertAlmostEqual(smoothed.volume, unsmoothed.volume, places=2)

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
        for sample_rate in (8000, 16000, 24000, 44100, 48000):
            with self.subTest(sample_rate=sample_rate):
                tracker = AudioVolumeTracker(smoothing_factor=1.0)
                feed(tracker, audio(sample_rate, amplitude=8000), sample_rate)
                self.assertGreater(tracker.volume, 0.0)


if __name__ == "__main__":
    unittest.main()
