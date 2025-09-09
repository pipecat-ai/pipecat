#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.string import match_endofsentence, parse_start_end_tags


class TestUtilsString(unittest.IsolatedAsyncioTestCase):
    async def test_endofsentence(self):
        assert match_endofsentence("This is a sentence.") == 19
        assert match_endofsentence("This is a sentence!") == 19
        assert match_endofsentence("This is a sentence?") == 19
        assert match_endofsentence("This is a sentence;") == 19
        assert match_endofsentence("This is a sentence...") == 21
        assert match_endofsentence("This is a sentence. This is another one") == 19
        assert match_endofsentence("This is for Mr. and Mrs. Jones.") == 31
        assert match_endofsentence("Meet the new Mr. and Mrs.") == 25
        assert match_endofsentence("U.S.A. and N.A.S.A.") == 19
        assert match_endofsentence("USA and NASA.") == 13
        assert match_endofsentence("My number is 123-456-7890.") == 26
        assert match_endofsentence("For information, call 411.") == 26
        assert match_endofsentence("My emails are foo@pipecat.ai and bar@pipecat.ai.") == 48
        assert match_endofsentence("My email is foo.bar@pipecat.ai.") == 31
        assert match_endofsentence("My email is spell(foo.bar@pipecat.ai).") == 38
        assert match_endofsentence("My email is <spell>foo.bar@pipecat.ai</spell>.") == 46
        assert match_endofsentence("The number pi is 3.14159.") == 25
        assert match_endofsentence("Valid scientific notation 1.23e4.") == 33
        assert match_endofsentence("Valid scientific notation 0.e4.") == 31
        assert match_endofsentence("It still early, it's 3:00 a.m.") == 30
        assert not match_endofsentence("This is not a sentence")
        assert not match_endofsentence("This is not a sentence,")
        assert not match_endofsentence("This is not a sentence, ")
        assert not match_endofsentence("Ok, Mr. Smith let's ")
        assert not match_endofsentence("Dr. Walker, I presume ")
        assert not match_endofsentence("Prof. Walker, I presume ")
        assert not match_endofsentence("zweitens, und 3")
        assert not match_endofsentence("Heute ist Dienstag, der 3")  # 3. Juli 2024
        assert not match_endofsentence("America, or the U.S")  # U.S.A.
        assert not match_endofsentence("My emails are foo@pipecat.ai and bar@pipecat.ai")
        assert not match_endofsentence("The number pi is 3.14159")

    async def test_endofsentence_multilingual(self):
        """Test sentence detection across various language families and scripts."""

        # Arabic script (Arabic, Urdu, Persian)
        arabic_sentences = [
            "مرحبا؟",  # Arabic question mark
            "السلام عليكم؛",  # Arabic semicolon
            "یہ اردو ہے۔",  # Urdu full stop
        ]
        for sentence in arabic_sentences:
            assert match_endofsentence(sentence), f"Failed for Arabic/Urdu: {sentence}"

        # Should not match incomplete Arabic
        assert not match_endofsentence("مرحبا،"), "Arabic comma should not end sentence"

        chinese_sentences = [
            "你好。",
            "你好！",
            "吃了吗？",
            "安全第一；",
        ]
        for sentence in chinese_sentences:
            assert match_endofsentence(sentence), f"Failed for Chinese: {sentence}"
        assert not match_endofsentence("你好，")

        hindi_sentences = [
            "हैलो।",
            "हैलो！",
            "आप खाये हैं？",
            "सुरक्षा पहले।",
        ]
        for sentence in hindi_sentences:
            assert match_endofsentence(sentence), f"Failed for Hindi: {sentence}"
        assert not match_endofsentence("हैलो，")

        # East Asian (Japanese, Korean)
        japanese_sentences = [
            "こんにちは。",  # Japanese
            "元気ですか？",  # Japanese question
            "ありがとう！",  # Japanese exclamation
        ]
        for sentence in japanese_sentences:
            assert match_endofsentence(sentence), f"Failed for Japanese: {sentence}"

        korean_sentences = [
            "안녕하세요。",  # Korean with ideographic period
            "어떻게 지내세요？",  # Korean question
        ]
        for sentence in korean_sentences:
            assert match_endofsentence(sentence), f"Failed for Korean: {sentence}"

        # Southeast Asian scripts
        thai_sentences = [
            "สวัสดี।",  # Thai with Devanagari-style punctuation
        ]
        for sentence in thai_sentences:
            assert match_endofsentence(sentence), f"Failed for Thai: {sentence}"

        myanmar_sentences = [
            "မင်္ဂလာပါ၊",  # Myanmar little section
            "ကျေးဇူးတင်ပါတယ်။",  # Myanmar section
        ]
        for sentence in myanmar_sentences:
            assert match_endofsentence(sentence), f"Failed for Myanmar: {sentence}"

        # Other Indic scripts (same punctuation as Hindi but different scripts)
        bengali_sentences = [
            "নমস্কার।",  # Bengali
            "আপনি কেমন আছেন？",  # Bengali question (uses Latin ?)
        ]
        for sentence in bengali_sentences:
            assert match_endofsentence(sentence), f"Failed for Bengali: {sentence}"

        tamil_sentences = [
            "வணக்கம்।",  # Tamil
            "நீங்கள் எப்படி இருக்கிறீர்கள்？",  # Tamil question
        ]
        for sentence in tamil_sentences:
            assert match_endofsentence(sentence), f"Failed for Tamil: {sentence}"

        # Armenian
        armenian_sentences = [
            "Բարև։",  # Armenian full stop
            "Ինչպես եք՞",  # Armenian question mark
            "Շնորհակալություն՜",  # Armenian exclamation
        ]
        for sentence in armenian_sentences:
            assert match_endofsentence(sentence), f"Failed for Armenian: {sentence}"

        # Ethiopic (Amharic)
        amharic_sentences = [
            "ሰላም።",  # Ethiopic full stop
            "እንዴት ነዎት፧",  # Ethiopic question mark
        ]
        for sentence in amharic_sentences:
            assert match_endofsentence(sentence), f"Failed for Amharic: {sentence}"

        # Languages using Latin punctuation (should still work)
        latin_script_sentences = [
            "Hola.",  # Spanish
            "Bonjour!",  # French
            "Guten Tag?",  # German
            "Привет.",  # Russian (Cyrillic but uses Latin punctuation)
            "Γεια σας.",  # Greek
            "שלום.",  # Hebrew
            "გამარჯობა.",  # Georgian
        ]
        for sentence in latin_script_sentences:
            assert match_endofsentence(sentence), f"Failed for Latin script: {sentence}"

    async def test_endofsentence_streaming_tokens(self):
        """Test the specific use case of streaming LLM tokens."""

        # These are the scenarios that were problematic with the original regex
        # Single tokens should not be considered complete sentences
        assert not match_endofsentence("Hello"), "Single token should not be sentence"
        assert not match_endofsentence("world"), "Single token should not be sentence"
        assert not match_endofsentence("The"), "Single token should not be sentence"
        assert not match_endofsentence("quick"), "Single token should not be sentence"

        # But accumulating tokens should eventually form sentences
        assert not match_endofsentence("Hello world"), "No punctuation = incomplete"
        assert match_endofsentence("Hello world.") == 12, "With punctuation = complete"

        # Test progressive building (simulating token streaming)
        tokens = ["The", " quick", " brown", " fox", " jumps", "."]
        accumulated = ""
        for i, token in enumerate(tokens):
            accumulated += token
            if i < len(tokens) - 1:  # All but the last token
                assert not match_endofsentence(accumulated), (
                    f"Should be incomplete at token {i}: '{accumulated}'"
                )
            else:  # Last token adds the period
                assert match_endofsentence(accumulated) == len(accumulated), (
                    f"Should be complete: '{accumulated}'"
                )

        # Test with multiple sentences
        assert match_endofsentence("First sentence. Second incomplete") == 15, (
            "Should return end of first sentence"
        )


class TestStartEndTags(unittest.IsolatedAsyncioTestCase):
    async def test_empty(self):
        assert parse_start_end_tags("", [], None, 0) == (None, 0)
        assert parse_start_end_tags("Hello from Pipecat!", [], None, 0) == (None, 0)

    async def test_simple(self):
        # (<a>, </a>)
        assert parse_start_end_tags("Hello from <a>Pipecat</a>!", [("<a>", "</a>")], None, 0) == (
            None,
            26,
        )
        assert parse_start_end_tags("Hello from <a>Pipecat", [("<a>", "</a>")], None, 0) == (
            ("<a>", "</a>"),
            21,
        )
        assert parse_start_end_tags("Hello from <a>Pipecat", [("<a>", "</a>")], None, 6) == (
            ("<a>", "</a>"),
            21,
        )

        # (spell(, ))
        assert parse_start_end_tags("Hello from spell(Pipecat)!", [("spell(", ")")], None, 0) == (
            None,
            26,
        )
        assert parse_start_end_tags("Hello from spell(Pipecat", [("spell(", ")")], None, 0) == (
            ("spell(", ")"),
            24,
        )

    async def test_multiple(self):
        # (<a>, </a>)
        assert parse_start_end_tags(
            "Hello from <a>Pipecat</a>! Hello <a>World</a>!", [("<a>", "</a>")], None, 0
        ) == (
            None,
            46,
        )

        assert parse_start_end_tags(
            "Hello from <a>Pipecat</a>! Hello <a>World", [("<a>", "</a>")], None, 0
        ) == (
            ("<a>", "</a>"),
            41,
        )
