#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import unittest
import wave

from pipecat.audio.utils import is_silence, pcm_to_wav


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


class TestIsSilence(unittest.TestCase):
    """An audio frame can arrive carrying no samples at all.

    Services build SpeechOutputAudioRawFrame straight from a transport's audio
    callback without checking the payload length, and the output transport calls
    is_silence() on every one of them. numpy's max() raises on an empty array, so
    a zero-length frame took down the speaking-detection path rather than being
    read as what it is: no audio, therefore no speech.
    """

    def test_empty_audio_is_silence(self):
        self.assertTrue(is_silence(b""))

    def test_quiet_audio_is_silence(self):
        pcm = (10).to_bytes(2, "little", signed=True) * 160
        self.assertTrue(is_silence(pcm))

    def test_loud_audio_is_not_silence(self):
        pcm = (3000).to_bytes(2, "little", signed=True) * 160
        self.assertFalse(is_silence(pcm))

    def test_a_single_loud_sample_is_not_silence(self):
        """max() is over the whole frame, so one loud sample is enough."""
        pcm = (10).to_bytes(2, "little", signed=True) * 159 + (3000).to_bytes(
            2, "little", signed=True
        )
        self.assertFalse(is_silence(pcm))

    def test_negative_amplitude_is_measured_by_magnitude(self):
        pcm = (-3000).to_bytes(2, "little", signed=True) * 160
        self.assertFalse(is_silence(pcm))
