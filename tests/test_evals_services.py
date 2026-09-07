#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval service constructors (config -> EvalJudge/EvalSpeech/EvalTranscriber)."""

import unittest

from pipecat.evals.judge import EvalJudge
from pipecat.evals.services import _cfg_language, cartesia_service
from pipecat.evals.speech import EvalSpeech, tts_cache_key, tts_sample_rate
from pipecat.evals.transcribe import EvalTranscriber
from pipecat.transcriptions.language import Language
from pipecat.utils.types import NOT_GIVEN


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

    def test_cache_key_distinguishes_language(self):
        # Two configs identical except for language must not collide, so an
        # English and a Chinese render of the same text get separate cache slots.
        self.assertNotEqual(
            tts_cache_key({"service": "cartesia", "voice": "v", "language": "en"}),
            tts_cache_key({"service": "cartesia", "voice": "v", "language": "zh"}),
        )
        # An absent language and an explicit empty one key to the same slot.
        self.assertEqual(
            tts_cache_key({"service": "cartesia", "voice": "v"}),
            tts_cache_key({"service": "cartesia", "voice": "v", "language": ""}),
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

    def test_language_reaches_cartesia_settings(self):
        # Cartesia is the one builder a unit test can construct: Whisper, Moonshine
        # and Kokoro load their models at construction time.
        service = cartesia_service(
            {"service": "cartesia", "voice": "v", "api_key": "test-key", "language": "zh"},
            16000,
        )
        self.assertEqual(service._settings.language, Language.ZH)

    def test_no_language_leaves_cartesia_default(self):
        # Omitting language must not force a value; the service keeps its own
        # default, which for Cartesia is Language.EN.
        service = cartesia_service(
            {"service": "cartesia", "voice": "v", "api_key": "test-key"}, 16000
        )
        self.assertEqual(service._settings.language, Language.EN)

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


class TestCfgLanguage(unittest.TestCase):
    def test_absent_leaves_the_field_unset(self):
        self.assertIs(_cfg_language({}), NOT_GIVEN)
        self.assertIs(_cfg_language({"language": None}), NOT_GIVEN)

    def test_blank_leaves_the_field_unset(self):
        # A key present but empty in the YAML means "unset", not "unknown language".
        self.assertIs(_cfg_language({"language": ""}), NOT_GIVEN)
        self.assertIs(_cfg_language({"language": "   "}), NOT_GIVEN)

    def test_code_or_language_accepted(self):
        self.assertEqual(_cfg_language({"language": "zh"}), Language.ZH)
        self.assertEqual(_cfg_language({"language": " zh-TW "}), Language.ZH_TW)
        self.assertEqual(_cfg_language({"language": Language.ES}), Language.ES)

    def test_unknown_code_rejected(self):
        with self.assertRaises(ValueError):
            _cfg_language({"language": "notalang"})

    def test_non_string_rejected(self):
        # YAML 1.1 reads a bare `language: no` as False rather than Norwegian, so
        # the coercion has to reject non-strings instead of passing them through.
        with self.assertRaises(ValueError):
            _cfg_language({"language": False})


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
