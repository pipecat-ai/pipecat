#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval service constructors (config -> EvalJudge/CachingTTSService/STT)."""

import unittest

from pipecat.evals.judge import EvalJudge
from pipecat.evals.services import (
    _cfg_language,
    cartesia_service,
    stt_service_from_config,
    tts_service_from_config,
)
from pipecat.evals.tts import CachingTTSService, tts_cache_key, tts_sample_rate
from pipecat.transcriptions.language import Language
from pipecat.utils.types import NOT_GIVEN


def _fake_stt(config):
    return ("FAKE_STT", config)


def _fake_tts(config):
    return ("FAKE_TTS", config)


def _fake_judge_llm(config):
    return ("FAKE_JUDGE", config)


class TestSTTServiceFromConfig(unittest.TestCase):
    def test_unknown_service_rejected(self):
        with self.assertRaises(ValueError):
            stt_service_from_config({"service": "nope"})

    def test_factory_escape_hatch(self):
        stt = stt_service_from_config({"factory": "tests.test_evals_services._fake_stt"})
        self.assertEqual(stt[0], "FAKE_STT")


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
            tts_service_from_config({"service": "nope", "voice": "v"})

    def test_missing_service_or_voice_rejected(self):
        with self.assertRaises(ValueError):
            tts_service_from_config({})

    def test_factory_escape_hatch(self):
        tts = tts_service_from_config({"factory": "tests.test_evals_services._fake_tts"})
        self.assertEqual(tts._inner[0], "FAKE_TTS")

    def test_language_reaches_cartesia_settings(self):
        # Cartesia is the one builder a unit test can construct: Whisper, Moonshine
        # and Kokoro load their models at construction time.
        service = cartesia_service(
            {"service": "cartesia", "voice": "v", "api_key": "test-key", "language": "zh"}
        )
        self.assertEqual(service._settings.language, Language.ZH)

    def test_no_language_leaves_cartesia_default(self):
        # Omitting language must not force a value; the service keeps its own
        # default, which for Cartesia is Language.EN.
        service = cartesia_service({"service": "cartesia", "voice": "v", "api_key": "test-key"})
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
            CachingTTSService(_FakeWS(), cache_key="k")


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


async def _run_tts_pcm(tts: CachingTTSService, text: str) -> bytes:
    """Drive ``run_tts`` and return the concatenated audio it yields."""
    from pipecat.frames.frames import TTSAudioRawFrame

    pcm = b""
    async for frame in tts.run_tts(text, "ctx"):
        if isinstance(frame, TTSAudioRawFrame):
            pcm += frame.audio
    return pcm


class TestCachingTTSCache(unittest.IsolatedAsyncioTestCase):
    async def test_cache_round_trip_and_sr_mismatch(self):
        import tempfile

        pcm = b"\x01\x02" * 1600  # 100ms of 16kHz mono

        with tempfile.TemporaryDirectory() as tmp:
            inner = _CountingTTS(pcm, 16000)
            tts = CachingTTSService(inner, cache_key="k", cache_dir=tmp)
            tts._sample_rate = 16000  # set by start(); skip the FrameProcessor lifecycle

            out = await _run_tts_pcm(tts, "hello")
            self.assertEqual(out, pcm)
            self.assertEqual(inner.calls, 1)

            # Second call hits the WAV cache; the inner service is not called again.
            out2 = await _run_tts_pcm(tts, "hello")
            self.assertEqual(out2, pcm)
            self.assertEqual(inner.calls, 1)

            # A different requested sample rate misses the cached file's rate and
            # regenerates (the cache slot is shared across rates by design).
            inner24 = _CountingTTS(pcm, 24000)
            tts24 = CachingTTSService(inner24, cache_key="k", cache_dir=tmp)
            tts24._sample_rate = 24000
            await _run_tts_pcm(tts24, "hello")
            self.assertEqual(inner24.calls, 1)


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
