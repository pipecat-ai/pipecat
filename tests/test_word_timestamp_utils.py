#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.word_timestamp_utils import merge_punct_tokens


class TestMergePunctTokens(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(merge_punct_tokens([]), [])

    def test_all_alnum_words_pass_through(self):
        input = [("hello", 0.0), ("world", 1.0)]
        self.assertEqual(merge_punct_tokens(input), [("hello", 0.0), ("world", 1.0)])

    def test_trailing_space_merged_into_preceding_word(self):
        input = [("I", 0.0), (" ", 0.2)]
        self.assertEqual(merge_punct_tokens(input), [("I ", 0.0)])

    def test_comma_space_merged_into_preceding_word(self):
        input = [("questions", 1.0), (", ", 1.2), ("explain", 1.4)]
        self.assertEqual(merge_punct_tokens(input), [("questions, ", 1.0), ("explain", 1.4)])

    def test_leading_space_with_no_preceding_word_discarded(self):
        input = [(" ", 0.0), ("hello", 0.5)]
        self.assertEqual(merge_punct_tokens(input), [("hello", 0.5)])

    def test_leading_empty_string_discarded(self):
        input = [("", 0.0), ("hello", 0.5)]
        self.assertEqual(merge_punct_tokens(input), [("hello", 0.5)])

    def test_multiple_consecutive_punct_tokens_merged(self):
        input = [("word", 0.0), (",", 0.1), (" ", 0.2), ("next", 0.3)]
        self.assertEqual(merge_punct_tokens(input), [("word, ", 0.0), ("next", 0.3)])

    def test_timestamp_of_preceding_word_is_kept(self):
        """Merged punct tokens adopt the preceding word's timestamp."""
        input = [("hello", 2.5), (",", 2.7)]
        result = merge_punct_tokens(input)
        self.assertEqual(result, [("hello,", 2.5)])

    def test_xml_tag_only_token_is_treated_as_punct(self):
        """A token that is only an XML tag (no alnum chars) merges into the preceding word."""
        input = [("word", 0.0), ("<break/>", 0.1), ("next", 0.3)]
        self.assertEqual(merge_punct_tokens(input), [("word<break/>", 0.0), ("next", 0.3)])

    def test_xml_tag_with_alnum_content_passes_through(self):
        """A token like '<spell>123</spell>' has alnum chars after stripping tags."""
        input = [("<spell>123</spell>", 0.0), ("and", 0.5)]
        self.assertEqual(merge_punct_tokens(input), [("<spell>123</spell>", 0.0), ("and", 0.5)])

    def test_inworld_style_full_stream(self):
        """Full Inworld-style raw stream produces expected merged output."""
        raw = [
            ("", 0.0),
            ("I", 0.1),
            (" ", 0.2),
            ("can", 0.3),
            (" ", 0.4),
            ("answer", 0.5),
            (" ", 0.6),
            ("questions", 0.7),
            (", ", 0.8),
            ("explain", 0.9),
            (" ", 1.0),
            ("things", 1.1),
            (".", 1.2),
        ]
        expected = [
            ("I ", 0.1),
            ("can ", 0.3),
            ("answer ", 0.5),
            ("questions, ", 0.7),
            ("explain ", 0.9),
            ("things.", 1.1),
        ]
        self.assertEqual(merge_punct_tokens(raw), expected)

    def test_only_punct_tokens_returns_empty(self):
        """A list containing only punct/space tokens produces an empty result."""
        input = [(" ", 0.0), (",", 0.1), (".", 0.2)]
        self.assertEqual(merge_punct_tokens(input), [])


if __name__ == "__main__":
    unittest.main()
