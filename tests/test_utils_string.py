#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.string import match_endofsentence, parse_eos_skip_tags


class TestUtilsString(unittest.IsolatedAsyncioTestCase):
    async def test_endofsentence(self):
        assert match_endofsentence("This is a sentence.") == 19
        assert match_endofsentence("This is a sentence!") == 19
        assert match_endofsentence("This is a sentence?") == 19
        assert match_endofsentence("This is a sentence;") == 19
        assert match_endofsentence("This is a sentence...") == 21
        assert match_endofsentence("This is a sentence . . .") == 24
        assert match_endofsentence("This is a sentence. ..") == 22
        assert match_endofsentence("This is for Mr. and Mrs. Jones.") == 31
        assert match_endofsentence("U.S.A and U.S.A..") == 17
        assert match_endofsentence("My emails are foo@pipecat.ai and bar@pipecat.ai.") == 48
        assert match_endofsentence("My email is foo.bar@pipecat.ai.") == 31
        assert match_endofsentence("My email is spell(foo.bar@pipecat.ai).") == 38
        assert match_endofsentence("My email is <spell>foo.bar@pipecat.ai</spell>.") == 46
        assert match_endofsentence("The number pi is 3.14159.") == 25
        assert match_endofsentence("Valid scientific notation 1.23e4.") == 33
        assert match_endofsentence("Valid scientific notation 0.e4.") == 31
        assert not match_endofsentence("This is not a sentence")
        assert not match_endofsentence("This is not a sentence,")
        assert not match_endofsentence("This is not a sentence, ")
        assert not match_endofsentence("Ok, Mr. Smith let's ")
        assert not match_endofsentence("Dr. Walker, I presume ")
        assert not match_endofsentence("Prof. Walker, I presume ")
        assert not match_endofsentence("zweitens, und 3.")
        assert not match_endofsentence("Heute ist Dienstag, der 3.")  # 3. Juli 2024
        assert not match_endofsentence("America, or the U.")  # U.S.A.
        assert not match_endofsentence("It still early, it's 3:00 a.")  # 3:00 a.m.
        assert not match_endofsentence("My emails are foo@pipecat.ai and bar@pipecat.ai")
        assert not match_endofsentence("The number pi is 3.14159")

    async def test_endofsentence_zh(self):
        chinese_sentences = [
            "你好。",
            "你好！",
            "吃了吗？",
            "安全第一；",
        ]
        for i in chinese_sentences:
            assert match_endofsentence(i)
        assert not match_endofsentence("你好，")

    async def test_endofsentence_hi(self):
        hindi_sentences = [
            "हैलो।",
            "हैलो！",
            "आप खाये हैं？",
            "सुरक्षा पहले।",
        ]
        for i in hindi_sentences:
            assert match_endofsentence(i)
        assert not match_endofsentence("हैलो，")


class TestEosSkipTags(unittest.IsolatedAsyncioTestCase):
    async def test_empty(self):
        assert parse_eos_skip_tags("", [], None, 0) == (0, None)
        assert parse_eos_skip_tags("Hello from Pipecat!", [], None, 0) == (0, None)

    async def test_simple(self):
        # (<a>, </a>)
        assert parse_eos_skip_tags("Hello from <a>Pipecat</a>!", [("<a>", "</a>")], None, 0) == (
            26,
            None,
        )
        assert parse_eos_skip_tags("Hello from <a>Pipecat", [("<a>", "</a>")], None, 0) == (
            21,
            ("<a>", "</a>"),
        )
        assert parse_eos_skip_tags("Hello from <a>Pipecat", [("<a>", "</a>")], None, 6) == (
            21,
            ("<a>", "</a>"),
        )

        # (spell(, ))
        assert parse_eos_skip_tags("Hello from spell(Pipecat)!", [("spell(", ")")], None, 0) == (
            26,
            None,
        )
        assert parse_eos_skip_tags("Hello from spell(Pipecat", [("spell(", ")")], None, 0) == (
            24,
            ("spell(", ")"),
        )

    async def test_multiple(self):
        # (<a>, </a>)
        assert parse_eos_skip_tags(
            "Hello from <a>Pipecat</a>! Hello <a>World</a>!", [("<a>", "</a>")], None, 0
        ) == (
            46,
            None,
        )

        assert parse_eos_skip_tags(
            "Hello from <a>Pipecat</a>! Hello <a>World", [("<a>", "</a>")], None, 0
        ) == (
            41,
            ("<a>", "</a>"),
        )
