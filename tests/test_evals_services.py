#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval service constructors (config -> EvalJudge/EvalSpeech/EvalTranscriber)."""

import unittest

from pipecat.evals.judge import EvalJudge
from pipecat.evals.speech import EvalSpeech, tts_cache_key, tts_sample_rate
from pipecat.evals.transcribe import EvalTranscriber


def _fake_stt(config, sample_rate):
    return ("FAKE_STT", config, sample_rate)


def _fake_tts(config, sample_rate):
    return ("FAKE_TTS", config, sample_rate)


def _fake_judge_llm(config):
    return ("FAKE_JUDGE", config)


class TestTranscriberFromConfig(unittest.TestCase):
    def test_whisper_default(self):
        from pipecat.services.whisper.stt import WhisperSTTService

        self.assertIsInstance(EvalTranscriber.from_config(None)._service, WhisperSTTService)
        self.assertIsInstance(
            EvalTranscriber.from_config({"model": "base"})._service, WhisperSTTService
        )

    def test_unknown_service_rejected(self):
        with self.assertRaises(ValueError):
            EvalTranscriber.from_config({"service": "nope"})

    def test_factory_escape_hatch(self):
        t = EvalTranscriber.from_config({"factory": "tests.test_evals_services._fake_stt"})
        self.assertEqual(t._service[0], "FAKE_STT")
        self.assertEqual(t._service[2], 16000)  # STT_SAMPLE_RATE


class TestVoiceFromConfig(unittest.TestCase):
    def test_cache_key_excludes_sample_rate(self):
        a = tts_cache_key({"service": "kokoro", "voice": "v", "model": "m", "sample_rate": 16000})
        b = tts_cache_key({"service": "kokoro", "voice": "v", "model": "m", "sample_rate": 24000})
        self.assertEqual(a, b)

    def test_cache_key_distinguishes_voice(self):
        self.assertNotEqual(
            tts_cache_key({"service": "kokoro", "voice": "a"}),
            tts_cache_key({"service": "kokoro", "voice": "b"}),
        )

    def test_sample_rate_default(self):
        self.assertEqual(tts_sample_rate({}), 16000)
        self.assertEqual(tts_sample_rate({"sample_rate": 24000}), 24000)

    def test_unknown_service_rejected(self):
        with self.assertRaises(ValueError):
            EvalSpeech.from_config({"service": "nope", "voice": "v"})

    def test_missing_service_or_voice_rejected(self):
        with self.assertRaises(ValueError):
            EvalSpeech.from_config({})

    def test_factory_escape_hatch(self):
        v = EvalSpeech.from_config(
            {"factory": "tests.test_evals_services._fake_tts", "sample_rate": 24000}
        )
        self.assertEqual(v._service[0], "FAKE_TTS")
        self.assertEqual(v._service[2], 24000)


class TestJudgeFromConfig(unittest.TestCase):
    def test_unknown_service_returns_none(self):
        self.assertIsNone(EvalJudge.from_config({"service": "nope"}))

    def test_factory_escape_hatch(self):
        j = EvalJudge.from_config({"factory": "tests.test_evals_services._fake_judge_llm"})
        self.assertIsNotNone(j)
        self.assertEqual(j._service[0], "FAKE_JUDGE")


if __name__ == "__main__":
    unittest.main()
