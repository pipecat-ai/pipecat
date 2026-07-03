#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.context.text_segment_map import TextSegmentMap


class TestTextSegmentMapBuild(unittest.TestCase):
    def test_equal_texts_produce_unchanged_segments(self):
        smap = TextSegmentMap("hello world", "hello world")
        for seg in smap._segments:
            self.assertFalse(seg.is_transformed)

    def test_currency_produces_transformed_segment(self):
        smap = TextSegmentMap(
            "forty two dollars and fifty cents",
            "$42.50",
        )
        transformed = [s for s in smap._segments if s.is_transformed]
        self.assertTrue(len(transformed) > 0)

    def test_segment_original_end_covers_full_text(self):
        original = "Your balance is $42.50"
        smap = TextSegmentMap(
            "Your balance is forty two dollars and fifty cents",
            original,
        )
        last = smap._segments[-1]
        self.assertEqual(last.original_end, len(original))

    def test_unchanged_prefix_segment(self):
        smap = TextSegmentMap(
            "Your balance is forty two dollars",
            "Your balance is $42",
        )
        first = smap._segments[0]
        self.assertFalse(first.is_transformed)
        self.assertEqual(first.original, "Your balance is ")

    def test_tts_alnum_count_correct(self):
        smap = TextSegmentMap("forty two", "$42")
        seg = next(s for s in smap._segments if s.is_transformed)
        self.assertEqual(seg.tts_alnum_count, len("fortytwo"))

    def test_original_alnum_count_correct(self):
        smap = TextSegmentMap("fifty percent", "50%")
        seg = next(s for s in smap._segments if s.is_transformed)
        self.assertEqual(seg.original_alnum_count, 2)  # "50"


class TestTextSegmentMapAdvance(unittest.TestCase):
    def _make_currency_map(self):
        return TextSegmentMap(
            "Your balance is forty two dollars and fifty cents",
            "Your balance is $42.50",
        )

    def test_unchanged_words_advance_user_facing_pos(self):
        smap = self._make_currency_map()
        smap.advance(4)  # Your
        self.assertGreater(smap.user_facing_pos, 0)
        self.assertFalse(smap.in_transformed_segment)

    def test_in_transformed_segment_true_mid_segment(self):
        smap = self._make_currency_map()
        smap.advance(4)  # Your
        smap.advance(7)  # balance
        smap.advance(2)  # is
        smap.advance(5)  # forty — enters transformed segment
        self.assertTrue(smap.in_transformed_segment)

    def test_cursors_held_during_transformed_segment(self):
        smap = self._make_currency_map()
        smap.advance(4)
        smap.advance(7)
        smap.advance(2)
        pos_before = smap.user_facing_pos
        smap.advance(5)  # forty
        smap.advance(3)  # two
        self.assertEqual(smap.user_facing_pos, pos_before)

    def test_cursors_jump_on_segment_completion(self):
        smap = self._make_currency_map()
        smap.advance(4)
        smap.advance(7)
        smap.advance(2)
        pos_before = smap.user_facing_pos
        smap.advance(5)  # forty
        smap.advance(3)  # two
        smap.advance(7)  # dollars
        smap.advance(3)  # and
        smap.advance(5)  # fifty
        smap.advance(5)  # cents — segment completes
        self.assertGreater(smap.user_facing_pos, pos_before)
        self.assertFalse(smap.in_transformed_segment)

    def test_last_completed_segment_on_transform_completion(self):
        smap = self._make_currency_map()
        smap.advance(4)
        smap.advance(7)
        smap.advance(2)
        smap.advance(5)
        smap.advance(3)
        smap.advance(7)
        smap.advance(3)
        smap.advance(5)
        smap.advance(5)  # cents
        seg = smap.last_completed_segment
        self.assertIsNotNone(seg)
        self.assertTrue(seg.is_transformed)
        self.assertIn("42", seg.original)

    def test_last_completed_segment_none_before_completion(self):
        smap = self._make_currency_map()
        smap.advance(4)  # Your
        self.assertIsNone(smap.last_completed_segment)

    def test_in_transformed_segment_false_before_segment(self):
        smap = self._make_currency_map()
        self.assertFalse(smap.in_transformed_segment)

    def test_in_transformed_segment_false_after_completion(self):
        smap = self._make_currency_map()
        smap.advance(4)
        smap.advance(7)
        smap.advance(2)
        smap.advance(5)
        smap.advance(3)
        smap.advance(7)
        smap.advance(3)
        smap.advance(5)
        smap.advance(5)
        self.assertFalse(smap.in_transformed_segment)


class TestTextSegmentMapWithLlmText(unittest.TestCase):
    def test_llm_pos_advances_past_digits_stops_before_closing_tag(self):
        # Transformed segment: "$42" → "forty two dollars" (15 alnum)
        # advance_by_alnums("<card>$42</card>", 0, 2) counts "4" and "2",
        # then the trailing loop hits "<" and stops — result is 9.
        smap = TextSegmentMap(
            "forty two dollars",
            "$42",
            llm_text="<card>$42</card>",
        )
        smap.advance(15)
        # Position 9 is the "<" that opens "</card>" — the two alnum digits
        # have been consumed but the closing tag is still unread (it will be
        # swept by WordCompletionTracker on the final "is_complete" word).
        self.assertEqual(smap.llm_pos, 9)

    def test_llm_pos_defaults_to_original_text_when_not_provided(self):
        # "50%" → "fifty percent" (12 alnum).  original_alnum_count = 2 ("50").
        # advance_by_alnums("50%", 0, 2) consumes "5" and "0", then the
        # trailing loop advances past "%" (non-alnum, non-space, non-tag) → 3.
        smap = TextSegmentMap("fifty percent", "50%")
        smap.advance(12)
        self.assertEqual(smap.llm_pos, 3)  # past "50%"


class TestTextSegmentMapReset(unittest.TestCase):
    def test_reset_restores_initial_state(self):
        smap = TextSegmentMap(
            "forty two dollars",
            "$42",
        )
        smap.advance(8)
        smap.reset()
        self.assertEqual(smap.user_facing_pos, 0)
        self.assertEqual(smap.llm_pos, 0)
        self.assertFalse(smap.in_transformed_segment)
        self.assertIsNone(smap.last_completed_segment)

    def test_reset_allows_replay(self):
        smap = TextSegmentMap("forty two dollars", "$42")
        smap.advance(8)
        pos_first = smap.user_facing_pos
        smap.reset()
        smap.advance(8)
        self.assertEqual(smap.user_facing_pos, pos_first)


class TestTextSegmentMapEqualTexts(unittest.TestCase):
    def test_all_segments_unchanged(self):
        smap = TextSegmentMap("hello world", "hello world")
        for seg in smap._segments:
            self.assertFalse(seg.is_transformed)

    def test_advance_works_for_equal_texts(self):
        smap = TextSegmentMap("hello world", "hello world")
        smap.advance(5)  # hello
        self.assertFalse(smap.in_transformed_segment)
        smap.advance(5)  # world
        self.assertFalse(smap.in_transformed_segment)


if __name__ == "__main__":
    unittest.main()
