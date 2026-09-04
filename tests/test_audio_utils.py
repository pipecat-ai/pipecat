#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import unittest
import wave

import numpy as np

from pipecat.audio.utils import _apply_half_hann_fade_out, pcm_to_wav


class TestHalfHannFadeOut(unittest.TestCase):
    def test_matches_descending_half_hann_and_pins_endpoints(self):
        samples = np.array([32767, -32768, 12000, -7000, 2000], dtype=np.int16)

        faded = np.frombuffer(_apply_half_hann_fade_out(samples.tobytes()), dtype=np.int16)

        phase = np.linspace(0.0, np.pi, num=len(samples), dtype=np.float64)
        gains = 0.5 * (1.0 + np.cos(phase))
        expected = np.rint(samples.astype(np.float64) * gains).astype(np.int16)
        expected[0] = samples[0]
        expected[-1] = 0
        np.testing.assert_array_equal(faded, expected)
        self.assertEqual(faded[0], samples[0])
        self.assertEqual(faded[-1], 0)

    def test_constant_signal_descends_monotonically_without_changing_length(self):
        samples = np.full(640, -32768, dtype=np.int16)

        faded_bytes = _apply_half_hann_fade_out(samples.tobytes())
        faded = np.frombuffer(faded_bytes, dtype=np.int16)

        self.assertEqual(len(faded_bytes), len(samples.tobytes()))
        self.assertTrue(np.all(np.diff(np.abs(faded.astype(np.int32))) <= 0))

    def test_handles_empty_and_single_sample_inputs(self):
        self.assertEqual(_apply_half_hann_fade_out(b""), b"")
        self.assertEqual(
            _apply_half_hann_fade_out(np.array([123], dtype=np.int16).tobytes()), b"\0\0"
        )

    def test_rejects_partial_sample(self):
        with self.assertRaisesRegex(ValueError, "complete 16-bit samples"):
            _apply_half_hann_fade_out(b"\x01")


class TestPcmToWav(unittest.TestCase):
    def _read_wav(self, data: bytes):
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            return (
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getframerate(),
                wav_file.readframes(wav_file.getnframes()),
            )

    def test_mono(self):
        pcm = b"\x01\x00" * 1600  # 0.1s of a constant sample at 16kHz
        wav = pcm_to_wav(pcm, 16000)
        num_channels, sample_width, sample_rate, frames = self._read_wav(wav)
        self.assertEqual(num_channels, 1)
        self.assertEqual(sample_width, 2)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(frames, pcm)

    def test_stereo(self):
        pcm = b"\x01\x00\x02\x00" * 2400  # 0.1s of interleaved stereo at 24kHz
        wav = pcm_to_wav(pcm, 24000, num_channels=2)
        num_channels, sample_width, sample_rate, frames = self._read_wav(wav)
        self.assertEqual(num_channels, 2)
        self.assertEqual(sample_width, 2)
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(frames, pcm)

    def test_empty(self):
        wav = pcm_to_wav(b"", 16000)
        num_channels, sample_width, sample_rate, frames = self._read_wav(wav)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(frames, b"")

    def test_bytearray(self):
        pcm = bytearray(b"\x01\x00" * 1600)
        wav = pcm_to_wav(pcm, 16000)
        _, _, _, frames = self._read_wav(wav)
        self.assertEqual(frames, bytes(pcm))

    def test_drops_partial_trailing_frame(self):
        pcm = b"\x01\x00\x02\x00" * 100 + b"\x03\x00"  # stereo plus a lone sample
        wav = pcm_to_wav(pcm, 24000, num_channels=2)
        num_channels, _, _, frames = self._read_wav(wav)
        self.assertEqual(num_channels, 2)
        self.assertEqual(frames, pcm[:-2])


if __name__ == "__main__":
    unittest.main()
