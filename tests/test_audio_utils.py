#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import struct
import unittest
import wave

from pipecat.audio.utils import pcm_to_wav


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

    def test_typed_memoryview_preserves_complete_frames(self):
        for channels, samples in ((1, (1, -2, 3)), (2, (1, -2))):
            with self.subTest(channels=channels):
                pcm = struct.pack("<" + "h" * len(samples), *samples)
                wav = pcm_to_wav(memoryview(pcm).cast("h"), 16000, channels)
                num_channels, sample_width, sample_rate, frames = self._read_wav(wav)
                self.assertEqual(num_channels, channels)
                self.assertEqual(sample_width, 2)
                self.assertEqual(sample_rate, 16000)
                self.assertEqual(frames, pcm)

    def test_typed_memoryview_drops_only_partial_trailing_frame(self):
        pcm = struct.pack("<5h", 1, -2, 3, -4, 5)
        wav = pcm_to_wav(memoryview(pcm).cast("h"), 24000, num_channels=2)
        _, _, _, frames = self._read_wav(wav)
        self.assertEqual(frames, pcm[:-2])

    def test_byte_memoryview(self):
        pcm = b"\x01\x00\x02\x00\x03"
        wav = pcm_to_wav(memoryview(pcm), 16000)
        _, _, _, frames = self._read_wav(wav)
        self.assertEqual(frames, pcm[:-1])

    def test_drops_partial_trailing_frame(self):
        pcm = b"\x01\x00\x02\x00" * 100 + b"\x03\x00"  # stereo plus a lone sample
        wav = pcm_to_wav(pcm, 24000, num_channels=2)
        num_channels, _, _, frames = self._read_wav(wav)
        self.assertEqual(num_channels, 2)
        self.assertEqual(frames, pcm[:-2])


if __name__ == "__main__":
    unittest.main()
