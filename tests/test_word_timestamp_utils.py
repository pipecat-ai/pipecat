#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.word_timestamp_utils import merge_punct_tokens, split_trailing_number


class TestMergePunctTokens(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(merge_punct_tokens([]), [])

    def test_all_alnum_words_pass_through(self):
        input = [("hello", 0.0), ("world", 1.0)]
        self.assertEqual(merge_punct_tokens(input), [("hello", 0.0), ("world", 1.0)])

    def test_trailing_space_merged_and_stripped(self):
        input = [("I", 0.0), (" ", 0.2)]
        self.assertEqual(merge_punct_tokens(input), [("I", 0.0)])

    def test_comma_space_merged_and_stripped(self):
        input = [("questions", 1.0), (", ", 1.2), ("explain", 1.4)]
        self.assertEqual(merge_punct_tokens(input), [("questions,", 1.0), ("explain", 1.4)])

    def test_leading_space_with_no_preceding_word_discarded(self):
        input = [(" ", 0.0), ("hello", 0.5)]
        self.assertEqual(merge_punct_tokens(input), [("hello", 0.5)])

    def test_leading_empty_string_discarded(self):
        input = [("", 0.0), ("hello", 0.5)]
        self.assertEqual(merge_punct_tokens(input), [("hello", 0.5)])

    def test_multiple_consecutive_punct_tokens_merged_and_stripped(self):
        input = [("word", 0.0), (",", 0.1), (" ", 0.2), ("next", 0.3)]
        self.assertEqual(merge_punct_tokens(input), [("word,", 0.0), ("next", 0.3)])

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
        """Full Inworld-style raw stream produces expected merged and stripped output."""
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
            ("I", 0.1),
            ("can", 0.3),
            ("answer", 0.5),
            ("questions,", 0.7),
            ("explain", 0.9),
            ("things.", 1.1),
        ]
        self.assertEqual(merge_punct_tokens(raw), expected)

    def test_only_punct_tokens_returns_empty(self):
        """A list containing only punct/space tokens produces an empty result."""
        input = [(" ", 0.0), (",", 0.1), (".", 0.2)]
        self.assertEqual(merge_punct_tokens(input), [])

    def test_digit_tokens_of_one_number_are_merged(self):
        """Inworld reports the digits of a number as separate tokens with no space."""
        raw = [("have", 0.0), (" ", 0.1), ("2", 0.2), ("5", 0.3), ("0", 0.4), (" ", 0.5)]
        self.assertEqual(merge_punct_tokens(raw), [("have", 0.0), ("250", 0.2)])

    def test_separators_inside_a_number_are_kept(self):
        raw = [("1", 0.0), (",", 0.1), ("1", 0.2), ("9", 0.3), ("9", 0.4)]
        self.assertEqual(merge_punct_tokens(raw), [("1,199", 0.0)])
        raw = [("7", 0.0), (".", 0.1), ("5", 0.2), (" ", 0.3), ("hours", 0.4)]
        self.assertEqual(merge_punct_tokens(raw), [("7.5", 0.0), ("hours", 0.4)])

    def test_numbers_separated_by_whitespace_stay_apart(self):
        raw = [("2020", 0.0), (" ", 0.1), ("25", 0.2)]
        self.assertEqual(merge_punct_tokens(raw), [("2020", 0.0), ("25", 0.2)])
        raw = [("250", 0.0), (", ", 0.1), ("3", 0.2)]
        self.assertEqual(merge_punct_tokens(raw), [("250,", 0.0), ("3", 0.2)])

    def test_a_word_is_not_merged_into_a_preceding_number(self):
        raw = [("3", 0.0), ("rd", 0.1), (" ", 0.2), ("20", 0.3), ("26", 0.4)]
        self.assertEqual(merge_punct_tokens(raw), [("3", 0.0), ("rd", 0.1), ("2026", 0.3)])


class TestSplitTrailingNumber(unittest.TestCase):
    def test_trailing_number_is_held(self):
        raw = [("across", 0.0), (" ", 0.1), ("20", 0.2)]
        self.assertEqual(split_trailing_number(raw), ([("across", 0.0), (" ", 0.1)], [("20", 0.2)]))

    def test_trailing_number_with_separator_is_held(self):
        raw = [("is", 0.0), (" ", 0.1), ("1", 0.2), (",", 0.3)]
        self.assertEqual(
            split_trailing_number(raw), ([("is", 0.0), (" ", 0.1)], [("1", 0.2), (",", 0.3)])
        )

    def test_complete_words_are_not_held(self):
        for raw in (
            [("plans", 0.0)],
            [("250", 0.0), (" ", 0.1)],
            [("250", 0.0), (". ", 0.1)],
            [],
        ):
            self.assertEqual(split_trailing_number(raw), (raw, []))

    def test_held_tokens_merge_with_the_next_message(self):
        first, held = split_trailing_number([("across", 0.0), (" ", 0.1), ("20", 0.2)])
        self.assertEqual(
            merge_punct_tokens(first + held + [("26", 0.3), (" ", 0.4), ("plans", 0.5)]),
            [("across", 0.0), ("2026", 0.2), ("plans", 0.5)],
        )


if __name__ == "__main__":
    unittest.main()
