#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from collections.abc import AsyncGenerator
from contextlib import contextmanager

from loguru import logger

from pipecat.frames.frames import Frame, TTSSpeakFrame
from pipecat.services.cartesia.tts import CartesiaHttpTTSService, CartesiaTTSService
from pipecat.services.deepgram.flux.tts_base import DeepgramFluxTTSBase
from pipecat.services.deepgram.tts import DeepgramHttpTTSService, DeepgramTTSService
from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService, ElevenLabsTTSService
from pipecat.services.inworld.tts import InworldHttpTTSService, InworldTTSService
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.phonemes import ipa_phones, normalize_ipa, stress_before_vowels


@contextmanager
def captured_warnings():
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


class TestNormalizeIpa(unittest.TestCase):
    def test_strips_delimiters_and_syllable_breaks(self):
        self.assertEqual(normalize_ipa(" /ˈtʃɪ.kən/ "), "ˈtʃɪkən")
        self.assertEqual(normalize_ipa("[ˈtʃɪkən]"), "ˈtʃɪkən")

    def test_unifies_notation(self):
        self.assertEqual(normalize_ipa("ˈʧɪkən"), "ˈtʃɪkən")
        self.assertEqual(normalize_ipa("ˈt\u0361ʃɪkən"), "ˈtʃɪkən")
        self.assertEqual(normalize_ipa("ˈpiʦa"), "ˈpit\u0361sa")
        self.assertEqual(normalize_ipa("t\u035cs"), "t\u0361s")
        self.assertEqual(normalize_ipa("'ælɛgrə"), "ˈælɛɡrə")
        self.assertEqual(normalize_ipa("bi:"), "biː")

    def test_keeps_sounds(self):
        # r and ɹ, ɾ and t are different sounds in some languages; notation only.
        self.assertEqual(normalize_ipa("ˈbɛɾə ˈrɑk"), "ˈbɛɾə ˈrɑk")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_ipa("ˈɡlaɪ   mɛt"), "ˈɡlaɪ mɛt")


class TestIpaPhones(unittest.TestCase):
    def test_groups_multi_character_phones(self):
        self.assertEqual(ipa_phones("ˈtʃaɪnə"), ["ˈ", "tʃ", "aɪ", "n", "ə"])
        self.assertEqual(ipa_phones("ˈʧaɪnə"), ["ˈ", "tʃ", "aɪ", "n", "ə"])

    def test_groups_only_tied_clusters(self):
        self.assertEqual(ipa_phones("kæts"), ["k", "æ", "t", "s"])
        self.assertEqual(ipa_phones("ˈpit\u0361sa"), ["ˈ", "p", "i", "ts", "a"])

    def test_keeps_length_and_diacritics_with_their_phone(self):
        self.assertEqual(ipa_phones("biːn̩"), ["b", "iː", "n̩"])

    def test_stress_where_written(self):
        self.assertEqual(
            ipa_phones("mɛtˈfɔɹmɪn"), ["m", "ɛ", "t", "ˈ", "f", "ɔ", "ɹ", "m", "ɪ", "n"]
        )

    def test_stress_before_vowels(self):
        phones = stress_before_vowels(ipa_phones("ˌsɪpɹoʊˈflɑksəsɪn"))
        self.assertEqual("".join(phones), "sˌɪpɹoʊflˈɑksəsɪn")


class TestCartesiaPronunciation(unittest.TestCase):
    def test_ipa(self):
        self.assertEqual(
            CartesiaTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn"),
            "<<m|ɛ|t|f|ˈ|ɔ|ɹ|m|ɪ|n>>",
        )

    def test_notation_variants_format_the_same(self):
        a = CartesiaTTSService.format_pronunciation("Achoo", "/əˈʧu/")
        b = CartesiaTTSService.format_pronunciation("Achoo", "ə'tʃu")
        self.assertEqual(a, b)

    def test_phrase(self):
        self.assertEqual(
            CartesiaTTSService.format_pronunciation("Glyburide metformin", "ˈɡlaɪ mɛt"),
            "<<ɡ|l|ˈ|aɪ>> <<m|ɛ|t>>",
        )

    def test_http_service_matches(self):
        self.assertEqual(
            CartesiaHttpTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn"),
            CartesiaTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn"),
        )


class TestElevenLabsPronunciation(unittest.TestCase):
    def test_ipa(self):
        self.assertEqual(
            ElevenLabsTTSService.format_pronunciation("Allegra", "/əˈlɛgrə/"),
            '<phoneme alphabet="ipa" ph="əˈlɛɡrə">Allegra</phoneme>',
        )

    def test_phrase_gets_a_tag_per_word(self):
        self.assertEqual(
            ElevenLabsTTSService.format_pronunciation("Zovirax tablets", "ˈzoʊvəˌræks ˈtæbləts"),
            '<phoneme alphabet="ipa" ph="ˈzoʊvəˌræks">Zovirax</phoneme> '
            '<phoneme alphabet="ipa" ph="ˈtæbləts">tablets</phoneme>',
        )

    def test_word_count_mismatch(self):
        self.assertIsNone(
            ElevenLabsTTSService.format_pronunciation("Zovirax tablets", "ˈzoʊvəˌræks")
        )

    def test_escapes_markup(self):
        self.assertEqual(
            ElevenLabsTTSService.format_pronunciation("A&B", "eɪ"),
            '<phoneme alphabet="ipa" ph="eɪ">A&amp;B</phoneme>',
        )

    def test_http_service_matches(self):
        self.assertEqual(
            ElevenLabsHttpTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn"),
            ElevenLabsTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn"),
        )

    def test_dialogue_ipa(self):
        self.assertEqual(
            ElevenLabsDialogueTTSService.format_pronunciation("Metformin", "[mɛt'fɔɹmɪn]"),
            "/mɛtˈfɔɹmɪn/",
        )


class TestInworldPronunciation(unittest.TestCase):
    def test_ipa(self):
        self.assertEqual(InworldTTSService.format_pronunciation("Crete", "[kriːt]"), "/kriːt/")

    def test_notation_variants_format_the_same(self):
        a = InworldTTSService.format_pronunciation("Achoo", "/əˈʧu/")
        b = InworldTTSService.format_pronunciation("Achoo", "ə'tʃu")
        self.assertEqual(a, b)

    def test_unusable(self):
        fmt = InworldTTSService.format_pronunciation
        self.assertIsNone(fmt("Glyburide metformin", "ˈɡlaɪ mɛt"))  # one word per pair
        self.assertIsNone(fmt("Crete", ""))

    def test_http_service_matches(self):
        self.assertEqual(
            InworldHttpTTSService.format_pronunciation("Crete", "kriːt"),
            InworldTTSService.format_pronunciation("Crete", "kriːt"),
        )


class TestDeepgramPronunciation(unittest.TestCase):
    def test_ipa(self):
        self.assertEqual(
            DeepgramTTSService.format_pronunciation("dupilumab", "duːˈpɪljuːmæb"),
            '\\{"word": "dupilumab", "pronounce": "duːpˈɪljuːmæb"\\}',
        )

    def test_notation_variants_format_the_same(self):
        a = DeepgramTTSService.format_pronunciation("Achoo", "/əˈʧu/")
        b = DeepgramTTSService.format_pronunciation("Achoo", "ə'tʃu")
        self.assertEqual(a, b)

    def test_quotes_in_word_are_escaped(self):
        self.assertEqual(
            DeepgramTTSService.format_pronunciation('say "hi"', "haɪ"),
            '\\{"word": "say \\"hi\\"", "pronounce": "haɪ"\\}',
        )

    def test_unusable(self):
        fmt = DeepgramTTSService.format_pronunciation
        self.assertIsNone(fmt("Crete", ""))
        self.assertIsNone(fmt("ab", "ˈkʌɹənt" * 3))  # far longer than the word
        self.assertIsNone(fmt("dupilumab", "duːˈpɪljuːmæb" * 12))  # over 128 characters

    def test_short_word_gets_a_length_floor(self):
        self.assertIsNotNone(DeepgramTTSService.format_pronunciation("a", "əˈbaʊtðæt"))

    def test_http_service_matches(self):
        self.assertEqual(
            DeepgramHttpTTSService.format_pronunciation("Crete", "kriːt"),
            DeepgramTTSService.format_pronunciation("Crete", "kriːt"),
        )

    def test_flux_unsupported(self):
        self.assertIsNone(DeepgramFluxTTSBase.format_pronunciation("Crete", "kriːt"))


class TestPronunciationTransforms(unittest.IsolatedAsyncioTestCase):
    async def test_replaces_whole_words_case_insensitively(self):
        transform = ElevenLabsTTSService.pronunciation_transform_ipa({"metformin": "mɛtˈfɔɹmɪn"})
        result = await transform("Take METFORMIN twice. Metformin, then water.", "*")
        self.assertEqual(
            result,
            'Take <phoneme alphabet="ipa" ph="mɛtˈfɔɹmɪn">METFORMIN</phoneme> twice. '
            '<phoneme alphabet="ipa" ph="mɛtˈfɔɹmɪn">Metformin</phoneme>, then water.',
        )

    async def test_leaves_longer_and_hyphenated_words(self):
        transform = CartesiaTTSService.pronunciation_transform_ipa({"Metformin": "mɛtˈfɔɹmɪn"})
        text = "Metformins and glyburide-metformin and metformin-ER"
        self.assertEqual(await transform(text, "*"), text)

    async def test_longest_first(self):
        transform = ElevenLabsDialogueTTSService.pronunciation_transform_ipa(
            {"metformin": "mɛtˈfɔɹmɪn", "glyburide metformin": "ˈɡlaɪbjəˌraɪd mɛtˈfɔɹmɪn"}
        )
        result = await transform("Glyburide metformin, or metformin", "*")
        self.assertEqual(result, "/ˈɡlaɪbjəˌraɪd mɛtˈfɔɹmɪn/, or /mɛtˈfɔɹmɪn/")

    async def test_unusable_words_are_spoken_as_written(self):
        with captured_warnings() as messages:
            transform = InworldTTSService.pronunciation_transform_ipa(
                {"Xarelto": "zəˈɹɛltoʊ", "Glyburide metformin": "ˈɡlaɪ mɛt"}
            )
        text = "Glyburide metformin or Xarelto"
        self.assertEqual(await transform(text, "*"), "Glyburide metformin or /zəˈɹɛltoʊ/")
        self.assertTrue(any("Glyburide metformin" in m and "Xarelto" not in m for m in messages))

    async def test_base_service_supports_nothing(self):
        transform = TTSService.pronunciation_transform_ipa({"Metformin": "mɛtˈfɔɹmɪn"})
        self.assertEqual(await transform("Metformin", "*"), "Metformin")


class RecordingTTSService(TTSService):
    """Records the text it is asked to speak; pronunciations are wrapped in <>."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True, push_text_frames=False, stop_frame_timeout_s=0.1, **kwargs
        )
        self.spoken: list[str] = []
        self.pronunciations_supported = True

    @classmethod
    def format_pronunciation(cls, word: str, ipa: str) -> str | None:
        return f"<{ipa}>"

    @property
    def supports_pronunciations(self) -> bool:
        return self.pronunciations_supported

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.spoken.append(text.strip())
        if False:
            yield


async def _upper(text, aggregation_type):
    return text.upper()


def _speak(*texts: str) -> list[Frame]:
    """Speak each text, giving the service time to run before the next frame."""
    return [frame for text in texts for frame in (TTSSpeakFrame(text), SleepFrame(sleep=0.2))]


class TestPronunciationGating(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> RecordingTTSService:
        pronounce = RecordingTTSService.pronunciation_transform_ipa({"crete": "kriːt"})
        return RecordingTTSService(text_transforms=[("*", _upper), ("*", pronounce)])

    async def test_applied_when_available(self):
        tts = self._service()
        await run_test(tts, frames_to_send=_speak("Visit Crete"), start_timeout=5.0)
        self.assertEqual(tts.spoken, ["VISIT <kriːt>"])

    async def test_skipped_when_unavailable(self):
        tts = self._service()
        tts.pronunciations_supported = False
        with captured_warnings() as messages:
            await run_test(
                tts, frames_to_send=_speak("Visit Crete", "Crete again"), start_timeout=5.0
            )
        # Other transforms still run; the warning is logged once.
        self.assertEqual(tts.spoken, ["VISIT CRETE", "CRETE AGAIN"])
        self.assertEqual(sum("pronunciation markup" in m for m in messages), 1)


class TestElevenLabsPronunciationAvailability(unittest.TestCase):
    def _ws(self, model: str, ssml: bool | None) -> ElevenLabsTTSService:
        return ElevenLabsTTSService(
            api_key="test",
            settings=ElevenLabsTTSService.Settings(voice="v", model=model),
            enable_ssml_parsing=ssml,
        )

    def test_websocket_needs_phoneme_model_and_ssml_parsing(self):
        self.assertTrue(self._ws("eleven_flash_v2", True).supports_pronunciations)
        self.assertTrue(self._ws("eleven_turbo_v2", True).supports_pronunciations)
        self.assertFalse(self._ws("eleven_flash_v2_5", True).supports_pronunciations)
        self.assertFalse(self._ws("eleven_flash_v2", None).supports_pronunciations)
        self.assertFalse(self._ws("eleven_flash_v2", False).supports_pronunciations)

    def test_http_needs_phoneme_model(self):
        def http(model):
            return ElevenLabsHttpTTSService(
                api_key="test",
                aiohttp_session=None,
                settings=ElevenLabsHttpTTSService.Settings(voice="v", model=model),
            )

        self.assertTrue(http("eleven_turbo_v2").supports_pronunciations)
        self.assertFalse(http("eleven_multilingual_v2").supports_pronunciations)


if __name__ == "__main__":
    unittest.main()
