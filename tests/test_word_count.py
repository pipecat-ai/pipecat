#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.word_count import count_spoken_words


class TestCountSpokenWords(unittest.TestCase):
    def test_whitespace_separated_tokens(self):
        self.assertEqual(count_spoken_words("Hello there!"), 2)
        self.assertEqual(count_spoken_words("don't stop"), 2)
        self.assertEqual(count_spoken_words(""), 0)
        self.assertEqual(count_spoken_words("   "), 0)

    def test_each_cjk_character_counts(self):
        self.assertEqual(count_spoken_words("等一下，这辆车的价格是多少？"), 12)
        self.assertEqual(count_spoken_words("こんにちは"), 5)
        self.assertEqual(count_spoken_words("Hello 你好"), 3)
        self.assertEqual(count_spoken_words("100元"), 2)

    def test_hangul_stays_one_token(self):
        self.assertEqual(count_spoken_words("안녕하세요"), 1)
        self.assertEqual(count_spoken_words("안녕 하세요"), 2)
