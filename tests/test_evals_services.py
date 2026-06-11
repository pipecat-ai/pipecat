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
    def test_unknown_service_rejected(self):
        with self.assertRaises(ValueError):
            EvalTranscriber.from_config({"service": "nope"})

    def test_factory_escape_hatch(self):
        t = EvalTranscriber.from_config({"factory": "tests.test_evals_services._fake_stt"})
        self.assertEqual(t._service[0], "FAKE_STT")
        self.assertEqual(t._service[2], 16000)  # STT_SAMPLE_RATE

    def test_padding_secs(self):
        from pipecat.evals.transcribe import SILENCE_PAD_S

        default = EvalTranscriber.from_config({"factory": "tests.test_evals_services._fake_stt"})
        self.assertEqual(default._padding_secs, SILENCE_PAD_S)
        override = EvalTranscriber.from_config(
            {"factory": "tests.test_evals_services._fake_stt", "padding_secs": 0.5}
        )
        self.assertEqual(override._padding_secs, 0.5)


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

    def test_websocket_service_rejected(self):
        # run_tts can't be driven without a pipeline to manage the connection, so a
        # websocket-streaming TTS service must be rejected at construction.
        from pipecat.services.websocket_service import WebsocketService

        class _FakeWS(WebsocketService):
            async def _connect_websocket(self):
                pass

            async def _disconnect_websocket(self):
                pass

            async def _receive_messages(self):
                pass

        with self.assertRaises(ValueError):
            EvalSpeech(_FakeWS(), sample_rate=16000, cache_key="k")


class _CountingTTS:
    """Minimal stand-in for a TTSService: run_tts yields one audio frame."""

    def __init__(self, pcm: bytes, sample_rate: int):
        self.pcm = pcm
        self.sample_rate = sample_rate
        self.calls = 0

    async def run_tts(self, text, context_id):
        from pipecat.frames.frames import TTSAudioRawFrame

        self.calls += 1
        yield TTSAudioRawFrame(audio=self.pcm, sample_rate=self.sample_rate, num_channels=1)


class TestSpeechCache(unittest.IsolatedAsyncioTestCase):
    async def test_cache_round_trip_and_sr_mismatch(self):
        import tempfile

        pcm = b"\x01\x02" * 1600  # 100ms of 16kHz mono

        with tempfile.TemporaryDirectory() as tmp:
            tts = _CountingTTS(pcm, 16000)
            speech = EvalSpeech(tts, sample_rate=16000, cache_key="k", cache_dir=tmp)
            speech._started = True  # skip the FrameProcessor lifecycle

            out, sr = await speech.generate("hello")
            self.assertEqual((out, sr), (pcm, 16000))
            self.assertEqual(tts.calls, 1)

            # Second call hits the WAV cache; the service is not called again.
            out2, _ = await speech.generate("hello")
            self.assertEqual(out2, pcm)
            self.assertEqual(tts.calls, 1)

            # A different requested sample rate misses the cached file's rate and
            # regenerates (the cache slot is shared across rates by design).
            tts24 = _CountingTTS(pcm, 24000)
            speech24 = EvalSpeech(tts24, sample_rate=24000, cache_key="k", cache_dir=tmp)
            speech24._started = True
            await speech24.generate("hello")
            self.assertEqual(tts24.calls, 1)


class TestJudgeFromConfig(unittest.TestCase):
    def test_unknown_service_rejected(self):
        with self.assertRaises(ValueError):
            EvalJudge.from_config({"service": "nope"})

    def test_factory_escape_hatch(self):
        j = EvalJudge.from_config({"factory": "tests.test_evals_services._fake_judge_llm"})
        self.assertIsNotNone(j)
        self.assertEqual(j._service[0], "FAKE_JUDGE")


if __name__ == "__main__":
    unittest.main()
