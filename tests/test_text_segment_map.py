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
        smap.advance_word("Your")
        self.assertGreater(smap.user_facing_pos, 0)
        self.assertFalse(smap.in_transformed_segment)

    def test_in_transformed_segment_true_mid_segment(self):
        smap = self._make_currency_map()
        smap.advance_word("Your")
        smap.advance_word("balance")
        smap.advance_word("is")
        smap.advance_word("forty")  # enters transformed segment
        self.assertTrue(smap.in_transformed_segment)

    def test_cursors_held_during_transformed_segment(self):
        smap = self._make_currency_map()
        smap.advance_word("Your")
        smap.advance_word("balance")
        smap.advance_word("is")
        pos_before = smap.user_facing_pos
        smap.advance_word("forty")
        smap.advance_word("two")
        self.assertEqual(smap.user_facing_pos, pos_before)

    def test_cursors_jump_on_segment_completion(self):
        smap = self._make_currency_map()
        smap.advance_word("Your")
        smap.advance_word("balance")
        smap.advance_word("is")
        pos_before = smap.user_facing_pos
        smap.advance_word("forty")
        smap.advance_word("two")
        smap.advance_word("dollars")
        smap.advance_word("and")
        smap.advance_word("fifty")
        smap.advance_word("cents")  # segment completes
        self.assertGreater(smap.user_facing_pos, pos_before)
        self.assertFalse(smap.in_transformed_segment)

    def test_last_completed_segment_on_transform_completion(self):
        smap = self._make_currency_map()
        for word in ["Your", "balance", "is", "forty", "two", "dollars", "and", "fifty", "cents"]:
            smap.advance_word(word)
        seg = smap.last_completed_segment
        self.assertIsNotNone(seg)
        self.assertTrue(seg.is_transformed)
        self.assertIn("42", seg.original)

    def test_last_completed_segment_none_before_completion(self):
        smap = self._make_currency_map()
        smap.advance_word("Your")
        self.assertIsNone(smap.last_completed_segment)

    def test_in_transformed_segment_false_before_segment(self):
        smap = self._make_currency_map()
        self.assertFalse(smap.in_transformed_segment)

    def test_in_transformed_segment_false_after_completion(self):
        smap = self._make_currency_map()
        for word in ["Your", "balance", "is", "forty", "two", "dollars", "and", "fifty", "cents"]:
            smap.advance_word(word)
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
        smap.advance_word("forty")
        smap.advance_word("two")
        smap.advance_word("dollars")
        # Position 9 is the "<" that opens "</card>" — the two alnum digits
        # have been consumed but the closing tag is still unread (it will be
        # swept by WordCompletionTracker on the final "is_complete" word).
        self.assertEqual(smap.llm_pos, 9)

    def test_llm_pos_defaults_to_original_text_when_not_provided(self):
        # "50%" → "fifty percent" (12 alnum).  original_alnum_count = 2 ("50").
        # advance_by_alnums("50%", 0, 2) consumes "5" and "0", then the
        # trailing loop advances past "%" (non-alnum, non-space, non-tag) → 3.
        smap = TextSegmentMap("fifty percent", "50%")
        smap.advance_word("fifty")
        smap.advance_word("percent")
        self.assertEqual(smap.llm_pos, 3)  # past "50%"


class TestTextSegmentMapReset(unittest.TestCase):
    def test_reset_restores_initial_state(self):
        smap = TextSegmentMap(
            "forty two dollars",
            "$42",
        )
        smap.advance_word("forty")
        smap.advance_word("two")
        smap.reset()
        self.assertEqual(smap.user_facing_pos, 0)
        self.assertEqual(smap.llm_pos, 0)
        self.assertFalse(smap.in_transformed_segment)
        self.assertIsNone(smap.last_completed_segment)

    def test_reset_allows_replay(self):
        smap = TextSegmentMap("forty two dollars", "$42")
        smap.advance_word("forty")
        smap.advance_word("two")
        pos_first = smap.user_facing_pos
        smap.reset()
        smap.advance_word("forty")
        smap.advance_word("two")
        self.assertEqual(smap.user_facing_pos, pos_first)


class TestTextSegmentMapEqualTexts(unittest.TestCase):
    def test_all_segments_unchanged(self):
        smap = TextSegmentMap("hello world", "hello world")
        for seg in smap._segments:
            self.assertFalse(seg.is_transformed)

    def test_advance_works_for_equal_texts(self):
        smap = TextSegmentMap("hello world", "hello world")
        smap.advance_word("hello")
        self.assertFalse(smap.in_transformed_segment)
        smap.advance_word("world")
        self.assertFalse(smap.in_transformed_segment)


class TestTextSegmentMapTokenChangingReplacements(unittest.TestCase):
    """Whether segments are flagged as transformed when a replacement changes
    tokenization, versus when it only changes case or the connector between
    words.

    A replacement that splits one word into several changes the *word count*
    within the segment, which breaks the 1:1 token correspondence proportional
    advancement assumes -- it must be flagged transformed so the segment is
    held and committed atomically instead. A replacement that only changes
    case or swaps the connector between words (space vs. hyphen) keeps the
    same single-token structure, so proportional advancement still lands at
    the correct position; those are intentionally left unflagged here and are
    instead handled by lenient (case/connector-insensitive) span validation in
    ``WordCompletionTracker``.
    """

    def test_word_splitting_replacement_is_flagged_transformed(self):
        # "BODYPUMP" -> "body pump": same alnum content, different tokenization.
        smap = TextSegmentMap(
            "Try body pump on Monday morning.",
            "Try BODYPUMP on Monday morning.",
        )
        seg = next(s for s in smap._segments if s.original == "BODYPUMP")
        self.assertTrue(
            seg.is_transformed,
            "a replacement that splits one word into several must be treated as transformed",
        )

    def test_case_only_replacement_is_not_flagged_transformed(self):
        # "SQL" -> "sql": same alnum content, same single-token structure, only
        # case differs. Proportional advancement already lands correctly here.
        smap = TextSegmentMap(
            "Contact sql support today.",
            "Contact SQL support today.",
        )
        seg = next(s for s in smap._segments if s.original == "SQL")
        self.assertFalse(seg.is_transformed)

    def test_hyphenated_single_token_replacement_is_not_flagged_transformed(self):
        # "BODYPUMP" -> "body-pump": still a single token on both sides.
        smap = TextSegmentMap(
            "Try body-pump on Monday morning.",
            "Try BODYPUMP on Monday morning.",
        )
        seg = next(s for s in smap._segments if s.original == "BODYPUMP")
        self.assertFalse(seg.is_transformed)

    def test_different_length_replacement_is_already_flagged_transformed(self):
        # Control case: "HIIT" -> "hit" differs in alnum length, so it takes
        # the transformed/atomic path via the existing alnum-content check.
        smap = TextSegmentMap(
            "We run hit classes on Tuesday.",
            "We run HIIT classes on Tuesday.",
        )
        seg = next(s for s in smap._segments if s.original == "HIIT")
        self.assertTrue(seg.is_transformed)

    def test_acronym_letter_spacing_is_flagged_transformed(self):
        # "API" -> "A P I": same alnum content, but letter-spacing splits one
        # word into three -- the same word-count change as splitting replacements.
        smap = TextSegmentMap("A P I launched", "API launched")
        seg = next(s for s in smap._segments if s.original == "API")
        self.assertTrue(seg.is_transformed)


class TestTextSegmentMapSsmlPhonemeTag(unittest.TestCase):
    """SSML phoneme tags (e.g. ElevenLabs' <phoneme alphabet="ipa" ph="...">)
    wrap a word for pronunciation without changing its alnum content, but the
    surrounding markup means the segment must be treated as transformed (held
    atomically) rather than assumed to advance proportionally word-by-word.
    """

    def test_phoneme_wrapped_word_is_flagged_transformed(self):
        llm_text = "My name is Siobhan."
        tts_text = 'My name is <phoneme alphabet="ipa" ph="ʃəˈvɔːn">Siobhan</phoneme>.'
        smap = TextSegmentMap(tts_text, llm_text, llm_text)
        seg = next(s for s in smap._segments if "Siobhan" in s.original)
        self.assertTrue(seg.is_transformed)

    def test_in_transformed_segment_true_for_leading_zero_alnum_fragment(self):
        """Once the preceding segment is fully consumed, a fragment that itself
        contributes zero alnum chars (e.g. a still-open tag's attribute text,
        which normalizes to '') is textually already inside the transformed
        segment and must report in_transformed_segment=True. Otherwise callers
        (WordCompletionTracker.suppress_in_context) wrongly treat the fragment
        as outside any transform and try to attribute raw llm_text chars to it."""
        llm_text = "My name is Siobhan."
        tts_text = 'My name is <phoneme alphabet="ipa" ph="ʃəˈvɔːn">Siobhan</phoneme>.'
        smap = TextSegmentMap(tts_text, llm_text, llm_text)
        smap.advance_word("My")
        smap.advance_word("name")
        smap.advance_word("is")  # prior unchanged segment now fully consumed
        smap.advance_word("<phoneme")  # 0 alnum chars, but inside the transformed segment
        self.assertTrue(smap.in_transformed_segment)


if __name__ == "__main__":
    unittest.main()
