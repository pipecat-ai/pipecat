#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from contextlib import contextmanager

from loguru import logger

from pipecat.services.cartesia.tts import CartesiaHttpTTSService, CartesiaTTSService
from pipecat.services.deepgram.flux.tts_base import DeepgramFluxTTSBase
from pipecat.services.deepgram.tts import DeepgramHttpTTSService, DeepgramTTSService
from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService, ElevenLabsTTSService
from pipecat.services.inworld.tts import InworldHttpTTSService, InworldTTSService
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.phonemes import (
    PhonemeAlphabet,
    ipa_phones,
    ipa_to_arpabet,
    normalize_arpabet,
    normalize_ipa,
    stress_before_vowels,
)

IPA = PhonemeAlphabet.IPA
ARPABET = PhonemeAlphabet.ARPABET


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


class TestArpabet(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_arpabet(" m eh0 t  f ao1 r m ih0 n "), "M EH0 T F AO1 R M IH0 N")

    def test_single_vowel_word_gets_primary_stress(self):
        self.assertEqual(normalize_arpabet("K AE T"), "K AE1 T")

    def test_invalid(self):
        self.assertIsNone(normalize_arpabet("M EH T F AO1 R M IH N"))  # missing stress
        self.assertIsNone(normalize_arpabet("M1 EH0"))  # stress on a consonant
        self.assertIsNone(normalize_arpabet("QQ1"))
        self.assertIsNone(normalize_arpabet(""))

    def test_from_ipa(self):
        self.assertEqual(ipa_to_arpabet("mɛtˈfɔɹmɪn"), "M EH0 T F AO1 R M IH0 N")
        self.assertEqual(ipa_to_arpabet("ˈɡlaɪbjəˌraɪd"), "G L AY1 B Y AH0 R AY2 D")

    def test_from_ipa_r_coloured_vowel(self):
        self.assertEqual(ipa_to_arpabet("ˈnɜrs"), "N ER1 S")
        self.assertEqual(ipa_to_arpabet("ˈbɚd"), "B ER1 D")
        self.assertEqual(ipa_to_arpabet("ˈæspərɪn"), "AE1 S P ER0 IH0 N")
        self.assertEqual(ipa_to_arpabet("zəˈɹɛltoʊ"), "Z AH0 R EH1 L T OW0")
        self.assertEqual(ipa_to_arpabet("ˈkʌɹənt"), "K ER1 AH0 N T")
        self.assertEqual(ipa_to_arpabet("ˈkɑɹ"), "K AA1 R")

    def test_from_ipa_unknown_sound(self):
        self.assertIsNone(ipa_to_arpabet("ˈbax"))


class TestCartesiaPronunciation(unittest.TestCase):
    def test_ipa(self):
        self.assertEqual(
            CartesiaTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn", IPA),
            "<<m|ɛ|t|f|ˈ|ɔ|ɹ|m|ɪ|n>>",
        )

    def test_notation_variants_format_the_same(self):
        a = CartesiaTTSService.format_pronunciation("Achoo", "/əˈʧu/", IPA)
        b = CartesiaTTSService.format_pronunciation("Achoo", "ə'tʃu", IPA)
        self.assertEqual(a, b)

    def test_phrase(self):
        self.assertEqual(
            CartesiaTTSService.format_pronunciation("Glyburide metformin", "ˈɡlaɪ mɛt", IPA),
            "<<ɡ|l|ˈ|aɪ>> <<m|ɛ|t>>",
        )

    def test_arpabet_unsupported(self):
        self.assertIsNone(CartesiaTTSService.format_pronunciation("Cat", "K AE1 T", ARPABET))

    def test_http_service_matches(self):
        self.assertEqual(
            CartesiaHttpTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn", IPA),
            CartesiaTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn", IPA),
        )


class TestElevenLabsPronunciation(unittest.TestCase):
    def test_ipa(self):
        self.assertEqual(
            ElevenLabsTTSService.format_pronunciation("Allegra", "/əˈlɛgrə/", IPA),
            '<phoneme alphabet="ipa" ph="əˈlɛɡrə">Allegra</phoneme>',
        )

    def test_arpabet(self):
        self.assertEqual(
            ElevenLabsTTSService.format_pronunciation(
                "Metformin", "m eh0 t f ao1 r m ih0 n", ARPABET
            ),
            '<phoneme alphabet="cmu-arpabet" ph="M EH0 T F AO1 R M IH0 N">Metformin</phoneme>',
        )

    def test_phrase_gets_a_tag_per_word(self):
        self.assertEqual(
            ElevenLabsTTSService.format_pronunciation(
                "Zovirax tablets", "ˈzoʊvəˌræks ˈtæbləts", IPA
            ),
            '<phoneme alphabet="ipa" ph="ˈzoʊvəˌræks">Zovirax</phoneme> '
            '<phoneme alphabet="ipa" ph="ˈtæbləts">tablets</phoneme>',
        )

    def test_unusable(self):
        fmt = ElevenLabsTTSService.format_pronunciation
        self.assertIsNone(fmt("Zovirax tablets", "ˈzoʊvəˌræks", IPA))  # word count mismatch
        self.assertIsNone(fmt("Two words", "T UW1 W ER1 D Z", ARPABET))
        self.assertIsNone(fmt("Metformin", "M EH T F AO1", ARPABET))  # missing stress

    def test_escapes_markup(self):
        self.assertEqual(
            ElevenLabsTTSService.format_pronunciation("A&B", "eɪ", IPA),
            '<phoneme alphabet="ipa" ph="eɪ">A&amp;B</phoneme>',
        )

    def test_http_service_matches(self):
        self.assertEqual(
            ElevenLabsHttpTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn", IPA),
            ElevenLabsTTSService.format_pronunciation("Metformin", "mɛtˈfɔɹmɪn", IPA),
        )

    def test_dialogue_ipa(self):
        self.assertEqual(
            ElevenLabsDialogueTTSService.format_pronunciation("Metformin", "[mɛt'fɔɹmɪn]", IPA),
            "/mɛtˈfɔɹmɪn/",
        )

    def test_dialogue_arpabet_unsupported(self):
        self.assertIsNone(
            ElevenLabsDialogueTTSService.format_pronunciation("Cat", "K AE1 T", ARPABET)
        )


class TestInworldPronunciation(unittest.TestCase):
    def test_ipa(self):
        self.assertEqual(InworldTTSService.format_pronunciation("Crete", "[kriːt]", IPA), "/kriːt/")

    def test_notation_variants_format_the_same(self):
        a = InworldTTSService.format_pronunciation("Achoo", "/əˈʧu/", IPA)
        b = InworldTTSService.format_pronunciation("Achoo", "ə'tʃu", IPA)
        self.assertEqual(a, b)

    def test_unusable(self):
        fmt = InworldTTSService.format_pronunciation
        self.assertIsNone(fmt("Cat", "K AE1 T", ARPABET))
        self.assertIsNone(fmt("Glyburide metformin", "ˈɡlaɪ mɛt", IPA))  # one word per pair
        self.assertIsNone(fmt("Crete", "", IPA))

    def test_http_service_matches(self):
        self.assertEqual(
            InworldHttpTTSService.format_pronunciation("Crete", "kriːt", IPA),
            InworldTTSService.format_pronunciation("Crete", "kriːt", IPA),
        )


class TestDeepgramPronunciation(unittest.TestCase):
    def test_ipa(self):
        self.assertEqual(
            DeepgramTTSService.format_pronunciation("dupilumab", "duːˈpɪljuːmæb", IPA),
            '\\{"word": "dupilumab", "pronounce": "duːˈpɪljuːmæb"\\}',
        )

    def test_notation_variants_format_the_same(self):
        a = DeepgramTTSService.format_pronunciation("Achoo", "/əˈʧu/", IPA)
        b = DeepgramTTSService.format_pronunciation("Achoo", "ə'tʃu", IPA)
        self.assertEqual(a, b)

    def test_quotes_in_word_are_escaped(self):
        self.assertEqual(
            DeepgramTTSService.format_pronunciation('say "hi"', "haɪ", IPA),
            '\\{"word": "say \\"hi\\"", "pronounce": "haɪ"\\}',
        )

    def test_unusable(self):
        fmt = DeepgramTTSService.format_pronunciation
        self.assertIsNone(fmt("Cat", "K AE1 T", ARPABET))
        self.assertIsNone(fmt("Crete", "", IPA))
        self.assertIsNone(fmt("ab", "ˈkʌɹənt" * 3, IPA))  # far longer than the word
        self.assertIsNone(fmt("dupilumab", "duːˈpɪljuːmæb" * 12, IPA))  # over 128 characters

    def test_short_word_gets_a_length_floor(self):
        self.assertIsNotNone(DeepgramTTSService.format_pronunciation("a", "əˈbaʊtðæt", IPA))

    def test_http_service_matches(self):
        self.assertEqual(
            DeepgramHttpTTSService.format_pronunciation("Crete", "kriːt", IPA),
            DeepgramTTSService.format_pronunciation("Crete", "kriːt", IPA),
        )

    def test_flux_unsupported(self):
        self.assertIsNone(DeepgramFluxTTSBase.format_pronunciation("Crete", "kriːt", IPA))


class TestPronunciationTransforms(unittest.IsolatedAsyncioTestCase):
    async def test_replaces_whole_words_case_insensitively(self):
        transform = ElevenLabsTTSService.pronounce_ipa({"metformin": "mɛtˈfɔɹmɪn"})
        result = await transform("Take METFORMIN twice. Metformin, then water.", "*")
        self.assertEqual(
            result,
            'Take <phoneme alphabet="ipa" ph="mɛtˈfɔɹmɪn">METFORMIN</phoneme> twice. '
            '<phoneme alphabet="ipa" ph="mɛtˈfɔɹmɪn">Metformin</phoneme>, then water.',
        )

    async def test_leaves_longer_and_hyphenated_words(self):
        transform = CartesiaTTSService.pronounce_ipa({"Metformin": "mɛtˈfɔɹmɪn"})
        text = "Metformins and glyburide-metformin and metformin-ER"
        self.assertEqual(await transform(text, "*"), text)

    async def test_longest_first(self):
        transform = ElevenLabsDialogueTTSService.pronounce_ipa(
            {"metformin": "mɛtˈfɔɹmɪn", "glyburide metformin": "ˈɡlaɪbjəˌraɪd mɛtˈfɔɹmɪn"}
        )
        result = await transform("Glyburide metformin, or metformin", "*")
        self.assertEqual(result, "/ˈɡlaɪbjəˌraɪd mɛtˈfɔɹmɪn/, or /mɛtˈfɔɹmɪn/")

    async def test_arpabet(self):
        transform = ElevenLabsTTSService.pronounce_arpabet({"Xarelto": "Z AH0 R EH1 L T OW0"})
        self.assertEqual(
            await transform("Xarelto.", "*"),
            '<phoneme alphabet="cmu-arpabet" ph="Z AH0 R EH1 L T OW0">Xarelto</phoneme>.',
        )

    async def test_unsupported_words_are_spoken_as_written(self):
        with captured_warnings() as messages:
            transform = CartesiaTTSService.pronounce_arpabet({"Xarelto": "Z AH0 R EH1 L T OW0"})
        self.assertEqual(await transform("Xarelto", "*"), "Xarelto")
        self.assertTrue(any("Xarelto" in m for m in messages))

    async def test_base_service_supports_nothing(self):
        transform = TTSService.pronounce_ipa({"Metformin": "mɛtˈfɔɹmɪn"})
        self.assertEqual(await transform("Metformin", "*"), "Metformin")

    async def test_mixed_alphabets(self):
        ipa = ElevenLabsTTSService.pronounce_ipa({"Metformin": "mɛtˈfɔɹmɪn"})
        arpabet = ElevenLabsTTSService.pronounce_arpabet({"Xarelto": "Z AH0 R EH1 L T OW0"})
        text = await arpabet(await ipa("Metformin or Xarelto", "*"), "*")
        self.assertEqual(
            text,
            '<phoneme alphabet="ipa" ph="mɛtˈfɔɹmɪn">Metformin</phoneme> or '
            '<phoneme alphabet="cmu-arpabet" ph="Z AH0 R EH1 L T OW0">Xarelto</phoneme>',
        )


if __name__ == "__main__":
    unittest.main()
