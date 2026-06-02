#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval service builders (config -> service)."""

import unittest

from pipecat.evals.transcribe import build_stt_service
from pipecat.evals.voice import build_tts_service, tts_cache_key, tts_sample_rate


def _fake_stt(config, sample_rate):
    return ("FAKE_STT", config, sample_rate)


def _fake_tts(config, sample_rate):
    return ("FAKE_TTS", config, sample_rate)


class TestSTTBuilder(unittest.TestCase):
    def test_whisper_default(self):
        from pipecat.services.whisper.stt import WhisperSTTService

        self.assertIsInstance(build_stt_service(None), WhisperSTTService)
        self.assertIsInstance(build_stt_service({"model": "base"}), WhisperSTTService)

    def test_unknown_service_rejected(self):
        with self.assertRaises(ValueError):
            build_stt_service({"service": "nope"})

    def test_factory_escape_hatch(self):
        result = build_stt_service({"factory": "tests.test_evals_services._fake_stt"})
        self.assertEqual(result[0], "FAKE_STT")
        self.assertEqual(result[2], 16000)  # STT_SAMPLE_RATE


class TestTTSBuilder(unittest.TestCase):
    def test_cache_key_excludes_sample_rate(self):
        a = tts_cache_key({"service": "cartesia", "voice": "v", "model": "m", "sample_rate": 16000})
        b = tts_cache_key({"service": "cartesia", "voice": "v", "model": "m", "sample_rate": 24000})
        self.assertEqual(a, b)

    def test_cache_key_distinguishes_voice(self):
        self.assertNotEqual(
            tts_cache_key({"service": "cartesia", "voice": "a"}),
            tts_cache_key({"service": "cartesia", "voice": "b"}),
        )

    def test_sample_rate_default(self):
        self.assertEqual(tts_sample_rate({}), 16000)
        self.assertEqual(tts_sample_rate({"sample_rate": 24000}), 24000)

    def test_unknown_service_rejected(self):
        with self.assertRaises(ValueError):
            build_tts_service({"service": "nope", "voice": "v"}, 16000)

    def test_missing_service_or_voice_rejected(self):
        with self.assertRaises(ValueError):
            build_tts_service({}, 16000)

    def test_factory_escape_hatch(self):
        result = build_tts_service({"factory": "tests.test_evals_services._fake_tts"}, 24000)
        self.assertEqual(result[0], "FAKE_TTS")
        self.assertEqual(result[2], 24000)


if __name__ == "__main__":
    unittest.main()
