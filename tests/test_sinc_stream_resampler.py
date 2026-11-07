#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SincStreamAudioResampler."""

import unittest
from unittest.mock import patch

import numpy as np

from pipecat.audio.resamplers.sinc_stream_resampler import SincStreamAudioResampler

# Down and up, by small ratios and by the large ones 44.1 kHz needs.
RATE_PAIRS = [
    (24000, 8000),
    (24000, 16000),
    (48000, 8000),
    (44100, 16000),
    (8000, 24000),
    (16000, 24000),
    (16000, 44100),
]

AMPLITUDE = 16000


def noise(rate: int, secs: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(rate * secs)) * 4000).astype(np.int16)


def tone(freq: float, rate: int, secs: float = 1.0) -> np.ndarray:
    t = np.arange(int(rate * secs)) / rate
    return (np.sin(2 * np.pi * freq * t) * AMPLITUDE).astype(np.int16)


def chunk_sizes(total: int, size: int) -> list[int]:
    return [size] * (total // size)


def level_db(samples: np.ndarray, rate: int, freq: float) -> float:
    """Level of `freq` in `samples`, relative to the test tones' amplitude."""
    window = np.blackman(len(samples))
    spectrum = np.abs(np.fft.rfft(samples * window)) / (window.sum() / 2)
    k = int(np.argmin(np.abs(np.fft.rfftfreq(len(samples), 1 / rate) - freq)))
    return 20 * np.log10(max(spectrum[max(k - 2, 0) : k + 3].max(), 1e-12) / AMPLITUDE)


async def stream(
    resampler: SincStreamAudioResampler,
    samples: np.ndarray,
    sizes: list[int],
    in_rate: int,
    out_rate: int,
) -> list[np.ndarray]:
    chunks, position = [], 0
    for size in sizes:
        data = await resampler.resample(
            samples[position : position + size].tobytes(), in_rate, out_rate
        )
        chunks.append(np.frombuffer(data, dtype=np.int16))
        position += size
    return chunks


async def at_once(samples: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """The signal resampled in a single call."""
    resampler = SincStreamAudioResampler(clear_after_secs=None)
    data = await resampler.resample(samples.tobytes(), in_rate, out_rate)
    return np.frombuffer(data, dtype=np.int16)


class TestSincStreamAudioResamplerOutput(unittest.IsolatedAsyncioTestCase):
    async def test_every_chunk_keeps_pace_with_its_input(self):
        for in_rate, out_rate in RATE_PAIRS:
            with self.subTest(in_rate=in_rate, out_rate=out_rate):
                resampler = SincStreamAudioResampler(clear_after_secs=None)
                sizes = chunk_sizes(in_rate, in_rate // 25)

                chunks = await stream(resampler, noise(in_rate), sizes, in_rate, out_rate)

                self.assertEqual({len(chunk) for chunk in chunks}, {out_rate // 25})

    async def test_streamed_output_matches_resampling_at_once(self):
        for in_rate, out_rate in RATE_PAIRS:
            with self.subTest(in_rate=in_rate, out_rate=out_rate):
                samples = noise(in_rate)
                resampler = SincStreamAudioResampler(clear_after_secs=None)

                streamed = np.concatenate(
                    await stream(
                        resampler, samples, chunk_sizes(in_rate, in_rate // 25), in_rate, out_rate
                    )
                )

                np.testing.assert_array_equal(streamed, await at_once(samples, in_rate, out_rate))

    async def test_any_chunk_sizes_give_the_same_output(self):
        rng = np.random.default_rng(1)
        for in_rate, out_rate in RATE_PAIRS:
            with self.subTest(in_rate=in_rate, out_rate=out_rate):
                samples = noise(in_rate)
                sizes = [int(size) for size in rng.integers(1, in_rate // 10, size=200)]
                sizes = sizes[: np.searchsorted(np.cumsum(sizes), len(samples), side="right")]
                resampler = SincStreamAudioResampler(clear_after_secs=None)

                streamed = np.concatenate(
                    await stream(resampler, samples, sizes, in_rate, out_rate)
                )

                expected = (await at_once(samples, in_rate, out_rate))[: len(streamed)]
                np.testing.assert_array_equal(streamed, expected)


class TestSincStreamAudioResamplerQuality(unittest.IsolatedAsyncioTestCase):
    async def _resampled_tone(self, freq: float, in_rate: int, out_rate: int) -> np.ndarray:
        resampler = SincStreamAudioResampler(clear_after_secs=None)
        sizes = chunk_sizes(in_rate, in_rate // 25)
        output = np.concatenate(
            await stream(resampler, tone(freq, in_rate), sizes, in_rate, out_rate)
        )
        # Skip the start, where the filter is still filling.
        return output[out_rate // 10 :].astype(np.float64)

    async def test_passband_is_kept(self):
        for in_rate, out_rate, freq in [
            (24000, 16000, 7000),
            (24000, 8000, 3400),
            (8000, 24000, 3400),
        ]:
            with self.subTest(in_rate=in_rate, out_rate=out_rate, freq=freq):
                output = await self._resampled_tone(freq, in_rate, out_rate)

                self.assertGreater(level_db(output, out_rate, freq), -0.1)

    async def test_downsampling_removes_what_would_alias(self):
        # A tone above the new Nyquist frequency would fold back to out_rate - freq.
        for in_rate, out_rate, freq in [(24000, 16000, 10000), (24000, 8000, 5000)]:
            with self.subTest(in_rate=in_rate, out_rate=out_rate, freq=freq):
                output = await self._resampled_tone(freq, in_rate, out_rate)

                self.assertLess(level_db(output, out_rate, out_rate - freq), -80)

    async def test_upsampling_removes_images(self):
        # Upsampling mirrors a tone around the old Nyquist frequency, to in_rate - freq.
        for in_rate, out_rate, freq in [(8000, 24000, 3000), (16000, 48000, 7000)]:
            with self.subTest(in_rate=in_rate, out_rate=out_rate, freq=freq):
                output = await self._resampled_tone(freq, in_rate, out_rate)

                self.assertLess(level_db(output, out_rate, in_rate - freq), -80)


class TestSincStreamAudioResamplerLifecycle(unittest.IsolatedAsyncioTestCase):
    CHUNK = noise(24000, 0.04)

    async def _fresh_output(self) -> bytes:
        return await SincStreamAudioResampler().resample(self.CHUNK.tobytes(), 24000, 8000)

    async def test_equal_rates_pass_audio_through(self):
        data = self.CHUNK.tobytes()

        self.assertIs(await SincStreamAudioResampler().resample(data, 16000, 16000), data)

    async def test_empty_input_gives_empty_output(self):
        self.assertEqual(await SincStreamAudioResampler().resample(b"", 24000, 8000), b"")

    async def test_other_rates_are_refused_once_set_up(self):
        resampler = SincStreamAudioResampler()
        await resampler.resample(self.CHUNK.tobytes(), 24000, 8000)

        with self.assertRaises(ValueError):
            await resampler.resample(self.CHUNK.tobytes(), 24000, 16000)

    async def test_flush_emits_the_delayed_end_and_starts_over(self):
        resampler = SincStreamAudioResampler(clear_after_secs=None)
        await stream(resampler, tone(1000, 24000), chunk_sizes(24000, 960), 24000, 8000)

        tail = np.frombuffer(await resampler.flush(), dtype=np.int16)

        self.assertTrue(len(tail))
        self.assertGreater(np.abs(tail).max(), AMPLITUDE / 2)
        self.assertEqual(
            await resampler.resample(self.CHUNK.tobytes(), 24000, 8000), await self._fresh_output()
        )

    async def test_flush_before_any_audio_is_empty(self):
        self.assertEqual(await SincStreamAudioResampler().flush(), b"")

    async def test_flush_without_new_audio_is_empty(self):
        resampler = SincStreamAudioResampler(clear_after_secs=None)
        await resampler.resample(self.CHUNK.tobytes(), 24000, 8000)
        self.assertTrue(await resampler.flush())

        self.assertEqual(await resampler.flush(), b"")

        await resampler.resample(self.CHUNK.tobytes(), 24000, 8000)
        await resampler.reset()

        self.assertEqual(await resampler.flush(), b"")

    async def test_flush_after_an_inactivity_clear_is_empty(self):
        resampler = SincStreamAudioResampler(clear_after_secs=0.2)
        with patch("pipecat.audio.resamplers.sinc_stream_resampler.time.time", return_value=100.0):
            await resampler.resample(self.CHUNK.tobytes(), 24000, 8000)
        with patch("pipecat.audio.resamplers.sinc_stream_resampler.time.time", return_value=100.5):
            await resampler.resample(b"", 24000, 8000)

        self.assertEqual(await resampler.flush(), b"")

    async def test_reset_discards_the_held_audio(self):
        resampler = SincStreamAudioResampler(clear_after_secs=None)
        await stream(resampler, tone(1000, 24000), chunk_sizes(24000, 960), 24000, 8000)

        await resampler.reset()

        self.assertEqual(
            await resampler.resample(self.CHUNK.tobytes(), 24000, 8000), await self._fresh_output()
        )

    async def test_inactivity_discards_the_held_audio(self):
        resampler = SincStreamAudioResampler(clear_after_secs=0.2)
        with patch("pipecat.audio.resamplers.sinc_stream_resampler.time.time", return_value=100.0):
            await stream(resampler, tone(1000, 24000), chunk_sizes(24000, 960), 24000, 8000)
        with patch("pipecat.audio.resamplers.sinc_stream_resampler.time.time", return_value=100.5):
            after_pause = await resampler.resample(self.CHUNK.tobytes(), 24000, 8000)

        self.assertEqual(after_pause, await self._fresh_output())

    async def test_a_short_pause_keeps_the_held_audio(self):
        resampler = SincStreamAudioResampler(clear_after_secs=0.2)
        with patch("pipecat.audio.resamplers.sinc_stream_resampler.time.time", return_value=100.0):
            await stream(resampler, tone(1000, 24000), chunk_sizes(24000, 960), 24000, 8000)
        with patch("pipecat.audio.resamplers.sinc_stream_resampler.time.time", return_value=100.1):
            after_pause = await resampler.resample(self.CHUNK.tobytes(), 24000, 8000)

        self.assertNotEqual(after_pause, await self._fresh_output())


if __name__ == "__main__":
    unittest.main()
