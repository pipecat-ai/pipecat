#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import hashlib
import unittest

import numpy as np

from pipecat.audio import g711, utils
from pipecat.audio.resamplers.sinc_resampler import SincAudioResampler

EVERY_SAMPLE = np.arange(-32768, 32768, dtype=np.int16).tobytes()
EVERY_CODE = bytes(range(256))


def pcm(*samples: int) -> bytes:
    return np.array(samples, dtype=np.int16).tobytes()


class TestConversions(unittest.TestCase):
    """Digests are of the standard library's ``audioop`` output for the same input."""

    def assertDigest(self, data: bytes, digest: str):
        self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_pcm_to_ulaw(self):
        self.assertEqual(g711.pcm_to_ulaw(pcm(0, 32767, -32768)), b"\xff\x80\x00")
        self.assertDigest(
            g711.pcm_to_ulaw(EVERY_SAMPLE),
            "81d633c9e6972a18c74a58720b96cb8ca0bdd096d4060b646dd708c3b846019a",
        )

    def test_ulaw_to_pcm(self):
        self.assertEqual(g711.ulaw_to_pcm(b"\xff\x80\x00"), pcm(0, 32124, -32124))
        self.assertDigest(
            g711.ulaw_to_pcm(EVERY_CODE),
            "3dab54339e520bb2c924826e3b72a917a2b612e9fd12fc867500f1d983a75827",
        )

    def test_pcm_to_alaw(self):
        self.assertEqual(g711.pcm_to_alaw(pcm(0, 32767, -32768)), b"\xd5\xaa\x2a")
        self.assertDigest(
            g711.pcm_to_alaw(EVERY_SAMPLE),
            "38488f6fd710f4686360edc4d38639f96c491595ef93f8eb8d62d5e07ca6ce7b",
        )

    def test_alaw_to_pcm(self):
        self.assertEqual(g711.alaw_to_pcm(b"\xd5\x55\x80\x00"), pcm(8, -8, 5504, -5504))
        self.assertDigest(
            g711.alaw_to_pcm(EVERY_CODE),
            "e04788d110e58ff8c70c93b8480190d973e3b67876b6119abbaec766cc75c174",
        )

    def test_empty_audio(self):
        for convert in (g711.pcm_to_ulaw, g711.ulaw_to_pcm, g711.pcm_to_alaw, g711.alaw_to_pcm):
            with self.subTest(convert=convert.__name__):
                self.assertEqual(convert(b""), b"")

    def test_pcm_must_be_whole_samples(self):
        for convert in (g711.pcm_to_ulaw, g711.pcm_to_alaw):
            with self.subTest(convert=convert.__name__):
                with self.assertRaises(ValueError):
                    convert(b"\x00\x00\x00")


class TestDeprecatedUtils(unittest.IsolatedAsyncioTestCase):
    async def test_encoders_resample_then_encode(self):
        audio = np.full(800, 1000, dtype=np.int16).tobytes()
        for encode in (utils.pcm_to_ulaw, utils.pcm_to_alaw):
            with self.subTest(encode=encode.__name__):
                with self.assertWarns(DeprecationWarning):
                    encoded = await encode(audio, 8000, 16000, SincAudioResampler())
                self.assertEqual(len(encoded), 1600)

    async def test_decoders_decode_then_resample(self):
        silence = {utils.ulaw_to_pcm: b"\xff" * 800, utils.alaw_to_pcm: b"\xd5" * 800}
        for decode, codes in silence.items():
            with self.subTest(decode=decode.__name__):
                with self.assertWarns(DeprecationWarning):
                    decoded = await decode(codes, 8000, 16000, SincAudioResampler())
                self.assertEqual(len(decoded), 3200)


if __name__ == "__main__":
    unittest.main()
