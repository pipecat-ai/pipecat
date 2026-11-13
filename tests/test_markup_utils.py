#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.text.alnum_utils import alnum_only, has_alnum
from pipecat.utils.text.markup_utils import (
    raw_offset_after_clean_chars,
    split_markup_runs,
    strip_complete_markup,
    strip_markup,
)


class TestStripMarkupHelpers(unittest.TestCase):
    """The markup-stripping primitives behind TextSegmentMap._markup_hop."""

    def test_strip_markup_removes_tags(self):
        self.assertEqual(strip_markup("<b>hi</b> there"), "hi there")

    def test_strip_markup_preserves_non_markup(self):
        self.assertEqual(strip_markup("1234-5678"), "1234-5678")

    def test_strip_markup_unclosed_tag_swallows_rest(self):
        # A '<' with no closing '>' consumes to the end (how a mid-tag fragment reads).
        self.assertEqual(strip_markup("keep <phoneme attr"), "keep ")

    def test_strip_markup_stray_gt_is_kept(self):
        self.assertEqual(strip_markup("a > b"), "a > b")

    def test_raw_len_maps_clean_prefix_to_raw_offset(self):
        # "hello" (5 clean chars) ends just before "</speak>" at raw index 12.
        self.assertEqual(raw_offset_after_clean_chars("<speak>hello</speak>", 5), 12)

    def test_raw_len_identity_without_markup(self):
        self.assertEqual(raw_offset_after_clean_chars("1234-5678", 9), 9)

    def test_raw_len_zero_or_negative_is_zero(self):
        self.assertEqual(raw_offset_after_clean_chars("<b>x</b>", 0), 0)

    def test_raw_len_beyond_available_returns_full_length(self):
        self.assertEqual(raw_offset_after_clean_chars("<b>x</b>", 99), len("<b>x</b>"))

    def test_raw_len_agrees_with_strip_markup(self):
        # Consuming len(strip_markup(t)) clean chars must land exactly at the raw
        # offset just past the last clean char: t[:pos] must strip down to the
        # same clean text (nothing missing), and t[pos] must be either past the
        # end of t or the start of trailing markup (nothing extra) -- the second
        # check matters because an implementation that overshoots a few chars
        # into a still-open trailing tag (short of reaching another clean char)
        # would still pass the first check alone, since strip_markup() truncates
        # an over-sliced, still-unclosed tag the same way either way.
        for t in ["<speak>hello</speak>", "1234-5678", "<a>x</a><b>y</b>", "plain"]:
            clean = strip_markup(t)
            pos = raw_offset_after_clean_chars(t, len(clean))
            self.assertEqual(strip_markup(t[:pos]), clean)
            self.assertTrue(pos == len(t) or t[pos] == "<")


class TestStripCompleteMarkupHelper(unittest.TestCase):
    """strip_complete_markup() is used on complete texts (TextSegment.is_transformed,
    WordCompletionTracker's default user_facing_text) where, unlike strip_markup(),
    a lone unmatched '<' is real content rather than a truncated tag."""

    def test_strip_complete_markup_removes_well_formed_tags(self):
        self.assertEqual(strip_complete_markup("<b>hi</b> there"), "hi there")

    def test_strip_complete_markup_keeps_unmatched_angle_bracket(self):
        self.assertEqual(strip_complete_markup("5 < 10"), "5 < 10")

    def test_strip_complete_markup_keeps_emoticon(self):
        self.assertEqual(strip_complete_markup("I love you <3 always"), "I love you <3 always")


class TestSplitMarkupRuns(unittest.TestCase):
    """split_markup_runs() gives a tag its own run so TextSegmentMap._build can
    give it its own segment, keeping the atomic span down to the tagged words."""

    def test_no_markup_yields_single_run(self):
        self.assertEqual(split_markup_runs("just plain text"), ["just plain text"])

    def test_empty_text_yields_no_runs(self):
        self.assertEqual(split_markup_runs(""), [])

    def test_tag_is_split_from_surrounding_text(self):
        self.assertEqual(
            split_markup_runs("I love to count <spell>1234</spell>."),
            ["I love to count ", "<spell>1234</spell>."],
        )

    def test_whitespace_inside_a_tag_does_not_split_it(self):
        self.assertEqual(
            split_markup_runs('say <phoneme alphabet="ipa">Siobhan</phoneme> now'),
            ["say ", '<phoneme alphabet="ipa">Siobhan</phoneme>', " now"],
        )

    def test_lone_angle_bracket_is_content(self):
        self.assertEqual(split_markup_runs("5 < 10 always"), ["5 < 10 always"])

    def test_runs_concatenate_back_to_the_input(self):
        for text in [
            "I love to count <spell>1234</spell>.",
            'say <phoneme alphabet="ipa">Siobhan</phoneme> now',
            "<break/>hello",
            "plain",
        ]:
            self.assertEqual("".join(split_markup_runs(text)), text)


class TestHasAlnum(unittest.TestCase):
    """has_alnum() is the predicate form of alnum_only(), markup included."""

    def test_agrees_with_alnum_only(self):
        for text in ["hello", "", "   ", "<break/>", "<b>hi</b>", "!!!", "😊", "5 < 10", "1234"]:
            self.assertEqual(has_alnum(text), bool(alnum_only(text)), text)

    def test_tag_name_is_not_content(self):
        self.assertFalse(has_alnum("<break/>"))


if __name__ == "__main__":
    unittest.main()
