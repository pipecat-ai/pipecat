#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.string import match_endofsentence


class TestUtilsString(unittest.IsolatedAsyncioTestCase):
    async def test_endofsentence(self):
        assert match_endofsentence("This is a sentence.")
        assert match_endofsentence("This is a sentence! ")
        assert match_endofsentence("This is a sentence?")
        assert match_endofsentence("This is a sentence:")
        assert match_endofsentence("This is a sentence;")
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

    async def test_endofsentence_zh(self):
        chinese_sentences = [
            "你好。",
            "你好！",
            "吃了吗？",
            "安全第一；",
            "他说：",
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
