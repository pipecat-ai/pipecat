#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.context.text_segment_map import TextSegmentMap, _HopKind
from pipecat.utils.text.alnum_utils import fold_for_matching


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


class TestTextSegmentMapStrayAngleBracket(unittest.TestCase):
    """A literal '<' with no matching '>' in ordinary TTS text (e.g. an emoticon
    like "<3" or a comparison like "5 < 10") is real content, not a truncated
    SSML tag, so it must not cause a segment to be misclassified as transformed."""

    def test_unchanged_segment_with_stray_angle_bracket_not_flagged_transformed(self):
        text = "I love you <3 always"
        smap = TextSegmentMap(text, text)
        seg = smap._segments[0]
        self.assertEqual(seg.tts, text)
        self.assertFalse(seg.is_transformed)


class TestClassifyHopLiteralMatchHandlesStrayAngleBracket(unittest.TestCase):
    """A literal '<3' arriving as its own word-timestamp token (e.g. an emoticon
    in ordinary text) is placed by _classify_hop's literal-matching strategies
    (1/2) directly, character for character against the segment's raw
    remaining text.
    """

    def test_literal_angle_bracket_word_placed_via_literal_strategy(self):
        hop = TextSegmentMap._classify_hop("<3 always", "<3")
        self.assertEqual(hop.kind, _HopKind.PLACED)
        # segment_advance == len(word) (offset 0 + len("<3")) is literal strategy's
        # formula; the markup-stripped strategy would compute this differently
        # (via raw_offset_after_clean_chars), so this pins down *which* strategy matched.
        self.assertEqual(hop.segment_advance, len("<3"))


class TestClassifyHopSkipsLeadingPunctuation(unittest.TestCase):
    """A word arriving right after punctuation the provider didn't repeat as its
    own token (e.g. the comma in "Yeah, I can") must still be placed -- the
    segment's leading punctuation run has to be skipped, not just its leading
    whitespace.
    """

    def test_word_after_comma_and_space_is_placed(self):
        hop = TextSegmentMap._classify_hop(", I can do that. ", "I")
        self.assertEqual(hop.kind, _HopKind.PLACED)
        self.assertEqual(hop.segment_advance, len(", I"))

    def test_full_sentence_advances_word_by_word(self):
        smap = TextSegmentMap("Yeah, I can do that.", "Yeah, I can do that.")
        for word in ("Yeah", "I", "can", "do"):
            self.assertTrue(smap.word_belongs_current_segment(word))
            smap.advance_word(word)
        self.assertTrue(smap.word_belongs_current_segment("that"))
        smap.advance_word("that")
        self.assertTrue(smap.is_complete)

    def test_tag_name_is_not_matched_as_a_spoken_word(self):
        """The punctuation skip must stop at '<' rather than scanning into a
        tag's name, or a tag name arriving as its own word-timestamp token would
        be PLACED as if it were spoken content and consume the tag.
        """
        hop = TextSegmentMap._classify_hop("<break/>hello", "break")
        self.assertNotEqual(
            hop.kind, _HopKind.PLACED, "'break' is the tag's name, not spoken content"
        )

    def test_word_after_leading_tag_still_placed_via_markup_strategy(self):
        """Stopping the punctuation skip at '<' must not cost the markup-stripped
        strategy (3) its match on a segment that opens with a tag.
        """
        hop = TextSegmentMap._classify_hop("<spell>4111 1111 1111 1111</spell>", "4111")
        self.assertEqual(hop.kind, _HopKind.PLACED)


class TestProviderTokenShapes(unittest.TestCase):
    """Word-timestamp tokens arrive in provider-specific shapes. Each shape below
    is handled by a different part of _classify_hop's matching, so each needs its
    own scenario.
    """

    def test_tokens_carrying_their_own_leading_whitespace(self):
        """Some providers include the separating space in the token (Inworld's
        " world"), so the match must succeed against the segment text as-is --
        before any leading-whitespace skip is applied.
        """
        smap = TextSegmentMap("Hello world", "Hello world")
        for word in ("Hello", " world"):
            self.assertTrue(smap.word_belongs_current_segment(word), f"{word!r} should belong")
            smap.advance_word(word)
        self.assertTrue(smap.is_complete)

    def test_tokens_uppercased_by_the_provider(self):
        """A provider that upper-cases its tokens needs the *word* side folded, not
        just the segment side -- the folded-candidate pass alone would still be
        comparing an upper-case word against lower-case source text.
        """
        smap = TextSegmentMap("hello world", "hello world")
        for word in ("HELLO", "WORLD"):
            self.assertTrue(smap.word_belongs_current_segment(word), f"{word!r} should belong")
            smap.advance_word(word)
        self.assertTrue(smap.is_complete)

    def test_tokens_carrying_an_accent_absent_from_the_source(self):
        """Accent folding must work in both directions: the provider may report a
        diacritic the source text doesn't have, not only strip one it does.
        """
        smap = TextSegmentMap("Visit the cafe today", "Visit the cafe today")
        for word in ("Visit", "the"):
            smap.advance_word(word)
        self.assertTrue(smap.word_belongs_current_segment("café"))


class TestClassifyHopCaseFoldRequiresWordBoundary(unittest.TestCase):
    """The case/accent-folded fallback strategy must not PLACE a word mid-word.

    Folding erases case before the prefix (startswith) match, so a short word
    that is only a case-insensitive prefix of a longer word (e.g. "account" vs
    "Accountant") must not be accepted -- that would silently corrupt the
    cursor by landing inside the longer word instead of at a real boundary.
    """

    def test_short_word_not_placed_inside_longer_word_via_case_fold(self):
        hop = TextSegmentMap._classify_hop(" Accountant", "account")
        self.assertNotEqual(
            hop.kind, _HopKind.PLACED, "must not match 'account' mid-word inside 'Accountant'"
        )

        smap = TextSegmentMap("Please talk to the Accountant", "Please talk to the Accountant")
        for word in ("Please", "talk", "to", "the"):
            smap.advance_word(word)
        self.assertFalse(smap.word_belongs_current_segment("account"))

    def test_whole_word_case_fold_still_matches_at_boundary(self):
        smap = TextSegmentMap("Please open the SQL database", "Please open the SQL database")
        for word in ("Please", "open", "the"):
            smap.advance_word(word)

        self.assertTrue(smap.word_belongs_current_segment("sql"))
        smap.advance_word("sql")
        self.assertTrue(smap.word_belongs_current_segment("database"))


class TestLeadingDuplicatePunctuation(unittest.TestCase):
    """A provider that reports a mark with the *following* word rather than the one
    it trails.

    The raw cursor stops before punctuation so the next token can still match it,
    while the LLM cursor sweeps it into the preceding word's span. A token that
    then leads with that same mark carries it a second time, and the map reports
    how much of its head to drop.
    """

    SENTENCE = "Yeah, I can do that."

    def _map_after_first_word(self):
        smap = TextSegmentMap(self.SENTENCE, self.SENTENCE, self.SENTENCE)
        smap.advance_word("Yeah")
        return smap

    def test_cursors_disagree_about_the_trailing_mark(self):
        smap = self._map_after_first_word()
        self.assertEqual(smap.raw_pos, 4, "raw cursor stops before the comma")
        self.assertEqual(smap.llm_pos, 5, "llm cursor swept the comma into 'Yeah'")

    def test_repeated_mark_is_reported_as_a_leading_duplicate(self):
        smap = self._map_after_first_word()
        smap.advance_word(", I")
        self.assertEqual(smap.last_leading_duplicate, 2, "drop the comma and its space")

    def test_word_without_the_mark_reports_nothing(self):
        smap = self._map_after_first_word()
        smap.advance_word("I")
        self.assertEqual(smap.last_leading_duplicate, 0)

    def test_punctuation_only_token_is_left_alone(self):
        """The mark arriving as its own event stands for this position itself."""
        smap = self._map_after_first_word()
        smap.advance_word(", ")
        self.assertEqual(smap.last_leading_duplicate, 0)

    def test_unconsumed_opening_punctuation_is_not_a_duplicate(self):
        sentence = 'He said "hello" today'
        smap = TextSegmentMap(sentence, sentence, sentence)
        smap.advance_word("He")
        smap.advance_word("said")
        smap.advance_word('"hello')
        self.assertEqual(smap.last_leading_duplicate, 0, "the quote is new content")

    def test_reset_clears_it(self):
        smap = self._map_after_first_word()
        smap.advance_word(", I")
        smap.reset()
        self.assertEqual(smap.last_leading_duplicate, 0)


class TestWordCarriesItsOwnPunctuation(unittest.TestCase):
    """A provider may punctuate a tagged span more than the source text does."""

    def test_hop_matches_a_word_whose_punctuation_the_span_lacks(self):
        """Cartesia reports "1234." for "<spell>1234</spell>\\n\\nHow ...", turning the
        line break into a sentence ending. The literal and folded strategies already
        retry with the word's trailing punctuation removed; the markup-stripped one
        has to as well, or the word matches nothing.
        """
        remaining = "<spell>1234</spell>\n\nHow can I help you today?"
        hop = TextSegmentMap._classify_hop(remaining, "1234.")
        self.assertEqual(hop.kind, _HopKind.PLACED)
        self.assertEqual(remaining[: hop.segment_advance], "<spell>1234")

    def test_sentence_tracks_through_the_extra_punctuation(self):
        text = "I love to count <spell>1234</spell>\n\nHow can I help you today?"
        smap = TextSegmentMap(text, text)
        for word in ["I", "love", "to", "count", "1234.", "How", "can", "I", "help", "you"]:
            smap.advance_word(word)
            self.assertIsNone(smap.last_overflow, f"{word!r} should not overflow")
        smap.advance_word("today?")
        self.assertTrue(smap.is_complete)
        self.assertEqual(smap.user_facing_pos, len(text))


class TestClassifyHopFoldsTypographicVariants(unittest.TestCase):
    """Strategy 2 folds typographic variants, not just case and accents.

    LLMs emit the typographic forms; a TTS service may report the ASCII form in its
    word-timestamp events (or the reverse). Without folding, the hop is NO_MATCH and the
    slot force-completes, collapsing the rest of the sentence into one frame.
    """

    def test_curly_apostrophe_matches_ascii_token(self):
        hop = TextSegmentMap._classify_hop("don’t worry", "don't")
        self.assertEqual(hop.kind, _HopKind.PLACED)

    def test_ascii_apostrophe_matches_curly_token(self):
        hop = TextSegmentMap._classify_hop("don't worry", "don’t")
        self.assertEqual(hop.kind, _HopKind.PLACED)

    def test_en_dash_matches_hyphen_token(self):
        hop = TextSegmentMap._classify_hop("2020–2021 report", "2020-2021")
        self.assertEqual(hop.kind, _HopKind.PLACED)

    def test_curly_quotes_match_straight_quotes(self):
        hop = TextSegmentMap._classify_hop("“hello” there", '"hello"')
        self.assertEqual(hop.kind, _HopKind.PLACED)

    def test_unrelated_token_still_does_not_match(self):
        """Folding must not turn a genuine mismatch into a match."""
        hop = TextSegmentMap._classify_hop("hello world", "goodbye")
        self.assertEqual(hop.kind, _HopKind.NO_MATCH)


class TestFoldPreservesLength(unittest.TestCase):
    """The fold must map each character to exactly one character.

    Strategy 2 finds a match offset in folded space and applies it to the raw text, so a
    fold that changed length would silently mis-place the cursor rather than fail loudly.
    The property is not self-evident: ``str.lower()`` is not length-preserving in Unicode
    (``"İ".lower()`` is two characters), and the fold survives that only because
    ``_fold_accented_char`` decomposes first and keeps the base character.
    """

    def test_fold_is_length_preserving_across_unicode(self):
        sample = "".join(chr(i) for i in range(32, 0x110000) if chr(i).isprintable())
        self.assertEqual(len(fold_for_matching(sample)), len(sample))

    def test_dotted_capital_i_folds_to_one_character(self):
        # "İ".lower() is "i" + COMBINING DOT ABOVE; folding must not inherit that.
        self.assertEqual(fold_for_matching("\u0130"), "i")

    def test_fold_is_narrow(self):
        """Only the listed variants change -- no blanket Unicode compatibility folding.

        A compatibility normalization would fold thousands of characters (CJK
        compatibility ideographs, halfwidth katakana, math alphanumerics) that no TTS
        service is known to substitute, widening the match surface far beyond the
        provider behaviour this is meant to absorb.
        """
        for char in ("\ufb01", "\u00bd", "\uff11", "\u4e09", "\u2032", "\u00a0"):
            self.assertEqual(fold_for_matching(char), char)


if __name__ == "__main__":
    unittest.main()
