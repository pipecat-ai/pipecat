#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.context.word_completion_tracker import WordCompletionTracker


class TestWordCompletionTrackerBasic(unittest.TestCase):
    def test_not_complete_before_any_words(self):
        tracker = WordCompletionTracker("Hello world")
        self.assertFalse(tracker.is_complete)

    def test_complete_after_all_words(self):
        tracker = WordCompletionTracker("Hello world")
        tracker.add_word_and_check_complete("Hello")
        self.assertFalse(tracker.is_complete)
        tracker.add_word_and_check_complete("world")
        self.assertTrue(tracker.is_complete)

    def test_add_word_and_check_complete_return_value(self):
        """add_word_and_check_complete returns True exactly when the tracker becomes complete."""
        tracker = WordCompletionTracker("Hi there")
        self.assertFalse(tracker.add_word_and_check_complete("Hi"))
        self.assertTrue(tracker.add_word_and_check_complete("there"))

    def test_single_word(self):
        tracker = WordCompletionTracker("Hello")
        self.assertTrue(tracker.add_word_and_check_complete("Hello"))

    def test_complete_stays_true_after_extra_words(self):
        """Extra words after completion force-complete (no-op remaining) and stay complete."""
        tracker = WordCompletionTracker("Hi")
        tracker.add_word_and_check_complete("Hi")
        result = tracker.add_word_and_check_complete("extra")
        self.assertTrue(result)
        self.assertTrue(tracker.is_complete)


class TestWordCompletionTrackerNormalization(unittest.TestCase):
    def test_punctuation_ignored_in_expected(self):
        """Punctuation in the source text is stripped before comparison."""
        tracker = WordCompletionTracker("Hello, world!")
        tracker.add_word_and_check_complete("Hello")
        tracker.add_word_and_check_complete("world")
        self.assertTrue(tracker.is_complete)

    def test_punctuation_ignored_in_words(self):
        """Punctuation attached to TTS word tokens is also stripped."""
        tracker = WordCompletionTracker("Hello world")
        tracker.add_word_and_check_complete("Hello,")
        tracker.add_word_and_check_complete("world!")
        self.assertTrue(tracker.is_complete)

    def test_case_insensitive(self):
        tracker = WordCompletionTracker("HELLO WORLD")
        tracker.add_word_and_check_complete("hello")
        tracker.add_word_and_check_complete("world")
        self.assertTrue(tracker.is_complete)

    def test_spaces_ignored(self):
        """Spaces in expected or received text do not count towards char totals."""
        tracker = WordCompletionTracker("a b c")
        # Normalized expected: "abc" (3 chars)
        tracker.add_word_and_check_complete("a")
        tracker.add_word_and_check_complete("b")
        self.assertFalse(tracker.is_complete)
        tracker.add_word_and_check_complete("c")
        self.assertTrue(tracker.is_complete)

    def test_numbers_kept(self):
        """Digits are treated as regular alphanumeric characters."""
        tracker = WordCompletionTracker("Room 42")
        tracker.add_word_and_check_complete("Room")
        self.assertFalse(tracker.is_complete)
        tracker.add_word_and_check_complete("42")
        self.assertTrue(tracker.is_complete)

    def test_mixed_punctuation_and_numbers(self):
        tracker = WordCompletionTracker("It costs $9.99!")
        # Normalized expected: "itcosts999"
        tracker.add_word_and_check_complete("It")
        tracker.add_word_and_check_complete("costs")
        self.assertFalse(tracker.is_complete)
        tracker.add_word_and_check_complete("$9.99!")
        self.assertTrue(tracker.is_complete)

    def test_special_characters_only_in_expected(self):
        """If the expected text normalizes to empty, the tracker is immediately complete."""
        tracker = WordCompletionTracker("...")
        self.assertTrue(tracker.is_complete)

    def test_emoji_only_expected_accepts_emoji_word_without_warning(self):
        """Emoji-only frame is already complete (normalizes to ''), but a TTS word event
        for the emoji must still be accepted gracefully and return True without a warning."""
        tracker = WordCompletionTracker("😊")
        self.assertTrue(tracker.is_complete)
        result = tracker.add_word_and_check_complete("😊")
        self.assertTrue(result)
        self.assertTrue(tracker.is_complete)

    def test_special_characters_only_in_word(self):
        """A word token that normalizes to empty contributes nothing."""
        tracker = WordCompletionTracker("hello")
        tracker.add_word_and_check_complete("---")  # normalizes to ""
        self.assertFalse(tracker.is_complete)
        tracker.add_word_and_check_complete("hello")
        self.assertTrue(tracker.is_complete)

    def test_ssml_tags_stripped_in_word(self):
        """SSML tags like <spell>...</spell> in TTS word tokens are stripped before comparison."""
        tracker = WordCompletionTracker("1234-5678")
        # Cartesia returns something like "<spell>1234-5678</spell>" as a word token
        self.assertTrue(tracker.add_word_and_check_complete("<spell>1234-5678</spell>"))

    def test_ssml_tags_do_not_inflate_char_count(self):
        """Tag names must not count as alphanumeric chars, preventing premature completion."""
        tracker = WordCompletionTracker("ab")
        # Without tag stripping "spell" (5 chars) would push received count over threshold.
        tracker.add_word_and_check_complete("<spell>a</spell>")
        self.assertFalse(tracker.is_complete)
        tracker.add_word_and_check_complete("b")
        self.assertTrue(tracker.is_complete)

    def test_ssml_tags_in_expected_text(self):
        """Tags in the expected text are also stripped."""
        tracker = WordCompletionTracker("<speak>hello</speak>")
        self.assertTrue(tracker.add_word_and_check_complete("hello"))


class TestWordCompletionTrackerReset(unittest.TestCase):
    def test_reset_clears_progress(self):
        tracker = WordCompletionTracker("Hello world")
        tracker.add_word_and_check_complete("Hello")
        tracker.add_word_and_check_complete("world")
        self.assertTrue(tracker.is_complete)
        tracker.reset()
        self.assertFalse(tracker.is_complete)

    def test_reset_allows_reuse(self):
        tracker = WordCompletionTracker("Hello world")
        tracker.add_word_and_check_complete("Hello")
        tracker.add_word_and_check_complete("world")
        tracker.reset()
        tracker.add_word_and_check_complete("Hello")
        self.assertFalse(tracker.is_complete)
        tracker.add_word_and_check_complete("world")
        self.assertTrue(tracker.is_complete)

    def test_reset_preserves_expected_text(self):
        """reset() only clears received chars; the expected text is unchanged."""
        tracker = WordCompletionTracker("Hi")
        tracker.add_word_and_check_complete("Hi")
        tracker.reset()
        # Re-adding the same word should complete again
        self.assertTrue(tracker.add_word_and_check_complete("Hi"))

    def test_reset_clears_raw_pos_cursor(self):
        """reset() resets the raw_text cursor so raw_consumed is correct after replay."""
        raw = "<card>4111</card>"
        tracker = WordCompletionTracker("4111", raw_text=raw)
        tracker.add_word_and_check_complete("4111")
        self.assertEqual(tracker.get_raw_consumed(), "<card>4111</card>")
        tracker.reset()
        tracker.add_word_and_check_complete("4111")
        # Cursor restarts from position 0 after reset.
        self.assertEqual(tracker.get_raw_consumed(), "<card>4111</card>")

    def test_reset_clears_expected_raw_pos_cursor(self):
        """reset() resets the expected_raw cursor so force-complete uses full text again."""
        tracker = WordCompletionTracker("number is")
        tracker.add_word_and_check_complete("number")
        # Partially advanced: remaining = " is"
        tracker.add_word_and_check_complete("4111")  # force-complete
        self.assertEqual(tracker.get_frame_word(), "is")

        tracker.reset()
        # After reset the cursor is back at 0, so force-complete sees the full text.
        tracker.add_word_and_check_complete("4111")
        self.assertEqual(tracker.get_frame_word(), "number is")


class TestWordCompletionTrackerEdgeCases(unittest.TestCase):
    def test_empty_expected_text(self):
        """An empty expected string is complete from the start."""
        tracker = WordCompletionTracker("")
        self.assertTrue(tracker.is_complete)

    def test_empty_word_adds_nothing(self):
        tracker = WordCompletionTracker("hello")
        tracker.add_word_and_check_complete("")
        self.assertFalse(tracker.is_complete)

    def test_partial_word_completion(self):
        """Chars accumulate; completion happens mid-add_word_and_check_complete if count is reached."""
        tracker = WordCompletionTracker("ab")
        # "ab" normalizes to 2 chars; one word covering both at once
        self.assertTrue(tracker.add_word_and_check_complete("ab"))

    def test_word_with_extra_chars_completes(self):
        """A single verbose token can satisfy a longer expected text."""
        tracker = WordCompletionTracker("Hi")
        self.assertTrue(tracker.add_word_and_check_complete("Hieveryone"))


class TestWordCompletionTrackerRealisticSentences(unittest.TestCase):
    # Sentence as it would appear in an AggregatedTextFrame from the LLM.
    SENTENCE = "You're welcome! If you have any more questions or need further examples, feel free to ask. Have a great day! 😊"

    # Words as a TTS word-timestamp service typically returns them:
    # punctuation attached to the adjacent word, emoji absent (unspeakable).
    TTS_WORDS = [
        "You're",
        "welcome!",
        "If",
        "you",
        "have",
        "any",
        "more",
        "questions",
        "or",
        "need",
        "further",
        "examples,",
        "feel",
        "free",
        "to",
        "ask.",
        "Have",
        "a",
        "great",
        "day!",
    ]

    def test_completes_after_all_tts_words(self):
        """Feeding every TTS word in order must complete the tracker."""
        tracker = WordCompletionTracker(self.SENTENCE)
        results = [tracker.add_word_and_check_complete(w) for w in self.TTS_WORDS]
        self.assertTrue(results[-1], "tracker should be complete after the last word")

    def test_not_complete_before_last_word(self):
        """Tracker must not report complete before the final word is added."""
        tracker = WordCompletionTracker(self.SENTENCE)
        for word in self.TTS_WORDS[:-1]:
            self.assertFalse(tracker.is_complete, f"should not be complete after '{word}'")

    def test_last_word_triggers_completion(self):
        """Only the last add_word_and_check_complete call should return True."""
        tracker = WordCompletionTracker(self.SENTENCE)
        intermediate = [tracker.add_word_and_check_complete(w) for w in self.TTS_WORDS[:-1]]
        self.assertFalse(any(intermediate))
        self.assertTrue(tracker.add_word_and_check_complete(self.TTS_WORDS[-1]))

    def test_emoji_in_expected_does_not_block_completion(self):
        """The 😊 at the end normalizes to '' so it adds no required chars."""
        tracker = WordCompletionTracker(self.SENTENCE)
        for word in self.TTS_WORDS:
            tracker.add_word_and_check_complete(word)
        self.assertTrue(tracker.is_complete)

    def test_chunked_delivery_two_words_per_call(self):
        """Some TTS providers return multiple words in one timestamp event."""
        tracker = WordCompletionTracker(self.SENTENCE)
        pairs = [
            self.TTS_WORDS[i] + " " + self.TTS_WORDS[i + 1]
            for i in range(0, len(self.TTS_WORDS) - 1, 2)
        ]
        # If word count is odd the last word is left alone.
        if len(self.TTS_WORDS) % 2:
            pairs.append(self.TTS_WORDS[-1])
        for chunk in pairs:
            tracker.add_word_and_check_complete(chunk)
        self.assertTrue(tracker.is_complete)

    def test_reset_and_replay(self):
        """After reset the same tracker can complete the same sentence again."""
        tracker = WordCompletionTracker(self.SENTENCE)
        for word in self.TTS_WORDS:
            tracker.add_word_and_check_complete(word)
        self.assertTrue(tracker.is_complete)

        tracker.reset()
        self.assertFalse(tracker.is_complete)

        for word in self.TTS_WORDS:
            tracker.add_word_and_check_complete(word)
        self.assertTrue(tracker.is_complete)

    def test_mid_sentence_is_not_complete(self):
        """Spot-check several points through the sentence."""
        tracker = WordCompletionTracker(self.SENTENCE)
        checkpoints = {4, 9, 14}  # after 5th, 10th, 15th word (0-indexed)
        for i, word in enumerate(self.TTS_WORDS[:-1]):
            tracker.add_word_and_check_complete(word)
            if i in checkpoints:
                self.assertFalse(
                    tracker.is_complete,
                    f"should not be complete after {i + 1} words",
                )

    def test_short_sentence_word_by_word(self):
        """Smaller realistic sentence: each word added individually."""
        sentence = "Of course! Here's a simple Python example."
        words = ["Of", "course!", "Here's", "a", "simple", "Python", "example."]
        tracker = WordCompletionTracker(sentence)
        for word in words[:-1]:
            self.assertFalse(tracker.add_word_and_check_complete(word))
        self.assertTrue(tracker.add_word_and_check_complete(words[-1]))

    def test_sentence_with_numbers(self):
        """Numbers are alphanumeric and must be counted in both directions."""
        sentence = "There are 3 options available for you."
        words = ["There", "are", "3", "options", "available", "for", "you."]
        tracker = WordCompletionTracker(sentence)
        for word in words[:-1]:
            self.assertFalse(tracker.add_word_and_check_complete(word))
        self.assertTrue(tracker.add_word_and_check_complete(words[-1]))

    def test_credit_card_sentence_with_raw_consumed(self):
        """Test that get_raw_consumed returns 'code:' after completing the sentence and all words."""
        sentence = "Here is a sample credit card number and a simple Python code:"
        words = [
            "Here",
            "is",
            "a",
            "samplecredit",
            "card",
            "number",
            "and",
            "a",
            "simple",
            "Python",
            "code:",
        ]

        # Provide raw_text parameter so get_raw_consumed() works
        tracker = WordCompletionTracker(sentence, raw_text=sentence)

        # Add all words except the last one
        for word in words[:-1]:
            result = tracker.add_word_and_check_complete(word)
            self.assertFalse(result, f"Should not be complete after adding '{word}'")

        # Add the final word - should complete the tracker
        result = tracker.add_word_and_check_complete(words[-1])
        self.assertTrue(result, "Should be complete after adding the last word")
        self.assertTrue(tracker.is_complete)

        # get_raw_consumed should return "code:" for the last word
        self.assertEqual(tracker.get_raw_consumed(), "code:")

    def test_maori_culture_sentence(self):
        """Test completion with Māori culture sentence - last word should be 'culture.'"""
        sentence = (
            "The indigenous Māori people are a significant part of the population and culture."
        )
        words = [
            "The",
            "indigenous",
            "Māori",
            "people",
            "are",
            "a",
            "significant",
            "part",
            "of",
            "the",
            "population",
            "and",
            "culture.",
        ]
        tracker = WordCompletionTracker(sentence, raw_text=sentence)

        # Add all words except the last one - should not complete
        for word in words[:-1]:
            result = tracker.add_word_and_check_complete(word)
            self.assertFalse(result, f"Should not be complete after adding '{word}'")

        # Add the final word "culture." - should complete the tracker
        result = tracker.add_word_and_check_complete(words[-1])
        self.assertTrue(result, "Should be complete after adding 'culture.'")
        self.assertTrue(tracker.is_complete)

        # get_raw_consumed should return "culture." for the last word
        self.assertEqual(tracker.get_frame_word(), "culture.")
        self.assertEqual(tracker.get_raw_consumed(), "culture.")

    def test_geography_sentence_frame_word_and_raw_consumed_validation(self):
        """Test geography sentence word by word, validating _frame_word and _raw_consumed match expected values.

        Sentence: 'Here are some key facts: **Geography:** - It consists mainly of two large islands:
        the North Island and the South Island, as well as many smaller islands.'

        This test validates that the received word, _frame_word, and _raw_consumed are exactly
        what we expect for each word addition, especially in special cases with punctuation.
        """
        sentence = "Here are some key facts: **Geography:** - It consists mainly of two large islands: the North Island and the South Island, as well as many smaller islands."
        raw_text = f"<geography>{sentence}</geography>"

        words = [
            "Here",
            "are",
            "some",
            "key",
            "facts:",
            "**Geography:**",
            "-",
            "It",
            "consists",
            "mainly",
            "of",
            "two",
            "large",
            "islands:",
            "the",
            "North",
            "Island",
            "and",
            "the",
            "South",
            "Island,",
            "as",
            "well",
            "as",
            "many",
            "smaller",
            "islands.",
        ]

        # Expected _frame_word for each word (should match the input word exactly)
        expected_frame_words = [
            "Here",
            "are",
            "some",
            "key",
            "facts:",
            "**Geography:**",
            "-",
            "It",
            "consists",
            "mainly",
            "of",
            "two",
            "large",
            "islands:",
            "the",
            "North",
            "Island",
            "and",
            "the",
            "South",
            "Island,",
            "as",
            "well",
            "as",
            "many",
            "smaller",
            "islands.",
        ]

        # Expected _raw_consumed for each word (spans from raw_text)
        expected_raw_consumed = [
            "<geography>Here",
            "are",
            "some",
            "key",
            "facts:",
            "**Geography:**",
            "-",
            "It",
            "consists",
            "mainly",
            "of",
            "two",
            "large",
            "islands:",
            "the",
            "North",
            "Island",
            "and",
            "the",
            "South",
            "Island,",
            "as",
            "well",
            "as",
            "many",
            "smaller",
            "islands.</geography>",
        ]

        tracker = WordCompletionTracker(sentence, raw_text=raw_text)

        for i, word in enumerate(words):
            is_complete = tracker.add_word_and_check_complete(word)

            # Test 1: Validate _frame_word matches expected
            actual_frame_word = tracker.get_frame_word()
            expected_frame_word = expected_frame_words[i]
            self.assertEqual(
                actual_frame_word,
                expected_frame_word,
                f"Word {i + 1} '{word}': expected _frame_word '{expected_frame_word}', got '{actual_frame_word}'",
            )

            # Test 2: Validate _raw_consumed matches expected
            actual_raw_consumed = tracker.get_raw_consumed()
            expected_raw = expected_raw_consumed[i]
            self.assertEqual(
                actual_raw_consumed,
                expected_raw,
                f"Word {i + 1} '{word}': expected _raw_consumed '{expected_raw}', got '{actual_raw_consumed}'",
            )

            # Test 3: Validate completion status
            if i == len(words) - 1:
                self.assertTrue(is_complete, f"Should be complete after final word '{word}'")
            else:
                self.assertFalse(
                    is_complete, f"Should not be complete after word '{word}' (position {i + 1})"
                )

        # Test special punctuation cases individually
        special_cases = [
            ("**Geography:**", "**Geography:**"),
            ("facts:", "facts:"),
            ("islands:", "islands:"),
            ("Island,", "Island,"),
            ("islands.", "islands."),
        ]

        for word, expected_frame in special_cases:
            tracker_special = WordCompletionTracker(word, raw_text=f"<test>{word}</test>")
            tracker_special.add_word_and_check_complete(word)

            actual_frame = tracker_special.get_frame_word()
            self.assertEqual(
                actual_frame,
                expected_frame,
                f"Special case '{word}': expected _frame_word '{expected_frame}', got '{actual_frame}'",
            )

            actual_raw = tracker_special.get_raw_consumed()
            expected_raw_special = f"<test>{word}</test>"
            self.assertEqual(
                actual_raw,
                expected_raw_special,
                f"Special case '{word}': expected _raw_consumed '{expected_raw_special}', got '{actual_raw}'",
            )


class TestWordCompletionTrackerWordBelongsHere(unittest.TestCase):
    def test_belongs_when_word_is_prefix_of_remaining(self):
        """word_belongs_here returns True when word alnum chars match the start of remaining."""
        tracker = WordCompletionTracker("4111 1111 1111 1111")
        self.assertTrue(tracker.word_belongs_here("4111"))

    def test_does_not_belong_when_word_mismatches_remaining(self):
        """word_belongs_here returns False when the word is clearly from a different slot."""
        tracker = WordCompletionTracker("number is")
        # '4' does not match 'n' (start of "numberis")
        self.assertFalse(tracker.word_belongs_here("4111"))

    def test_punctuation_only_word_is_neutral(self):
        """A word that normalizes to empty always belongs (no content to mismatch)."""
        tracker = WordCompletionTracker("hello")
        self.assertTrue(tracker.word_belongs_here("..."))
        self.assertTrue(tracker.word_belongs_here(","))
        self.assertTrue(tracker.word_belongs_here(""))

    def test_belongs_when_partial_prefix_matches(self):
        """Partial token (fewer chars than remaining) still passes if it is a prefix."""
        tracker = WordCompletionTracker("hello world")
        # remaining = "helloworld", word = "hel" — prefix matches
        self.assertTrue(tracker.word_belongs_here("hel"))

    def test_does_not_belong_when_no_remaining(self):
        """word_belongs_here returns False when the tracker is already complete."""
        tracker = WordCompletionTracker("hi")
        tracker.add_word_and_check_complete("hi")
        self.assertFalse(tracker.word_belongs_here("extra"))

    def test_belongs_advances_correctly_after_partial_consumption(self):
        """Remaining expected text shifts as words are consumed."""
        tracker = WordCompletionTracker("hello world")
        tracker.add_word_and_check_complete("hello")
        # remaining is now "world"
        self.assertTrue(tracker.word_belongs_here("world"))
        self.assertFalse(tracker.word_belongs_here("hello"))


class TestWordCompletionTrackerOverflow(unittest.TestCase):
    """Words that span the boundary of two AggregatedTextFrames."""

    def test_word_straddles_frame_boundary(self):
        """A word token that spans two frames splits at the alnum boundary."""
        tracker = WordCompletionTracker("hello")
        result = tracker.add_word_and_check_complete("helloworld")
        self.assertTrue(result)
        self.assertEqual(tracker.get_frame_word(), "hello")
        self.assertEqual(tracker.get_overflow(), "world")
        self.assertEqual(tracker.get_raw_overflow_word(), "world")

    def test_no_overflow_when_word_fits_exactly(self):
        """A word that exactly fills remaining slots produces no overflow."""
        tracker = WordCompletionTracker("hello")
        tracker.add_word_and_check_complete("hello")
        self.assertIsNone(tracker.get_overflow())
        self.assertIsNone(tracker.get_raw_overflow_word())
        self.assertEqual(tracker.get_frame_word(), "hello")

    def test_overflow_split_preserves_non_alnum_suffix(self):
        """The raw overflow word retains non-alnum chars after the split point."""
        tracker = WordCompletionTracker("1111")
        # "1111And" — 4 alnum chars for this frame, "And" as raw overflow
        result = tracker.add_word_and_check_complete("1111And")
        self.assertTrue(result)
        self.assertEqual(tracker.get_frame_word(), "1111")
        self.assertEqual(tracker.get_overflow(), "and")
        self.assertEqual(tracker.get_raw_overflow_word(), "And")

    def test_overflow_with_digits_splits_at_correct_position(self):
        """Split position is computed by alnum count, not byte offset."""
        tracker = WordCompletionTracker("4111")  # 4 alnum chars
        tracker.add_word_and_check_complete("41111111")
        self.assertEqual(tracker.get_frame_word(), "4111")
        self.assertEqual(tracker.get_overflow(), "1111")
        self.assertEqual(tracker.get_raw_overflow_word(), "1111")

    def test_overflow_flows_into_next_tracker(self):
        """Overflow word fed into the next tracker completes it correctly."""
        tracker1 = WordCompletionTracker("hello")
        tracker2 = WordCompletionTracker("world")

        tracker1.add_word_and_check_complete("helloworld")
        overflow = tracker1.get_raw_overflow_word()
        self.assertEqual(overflow, "world")

        result = tracker2.add_word_and_check_complete(overflow)
        self.assertTrue(result)
        self.assertEqual(tracker2.get_frame_word(), "world")
        self.assertIsNone(tracker2.get_overflow())

    def test_overflow_with_digits_flows_into_next_tracker(self):
        """Realistic card-number overflow: last token spans two frame slots."""
        tracker1 = WordCompletionTracker("4111 1111 1111 1111")
        tracker2 = WordCompletionTracker("And your")

        for word in ["4111", "1111", "1111"]:
            tracker1.add_word_and_check_complete(word)

        # "1111And" straddles the frame boundary
        result = tracker1.add_word_and_check_complete("1111And")
        self.assertTrue(result)
        self.assertEqual(tracker1.get_frame_word(), "1111")
        self.assertEqual(tracker1.get_raw_overflow_word(), "And")

        # Feed overflow into tracker2
        result = tracker2.add_word_and_check_complete(tracker1.get_raw_overflow_word())
        self.assertFalse(result)
        self.assertEqual(tracker2.get_frame_word(), "And")

        result = tracker2.add_word_and_check_complete("your")
        self.assertTrue(result)

    def test_no_overflow_state_on_normal_word(self):
        """After a normal (non-straddling) word, overflow getters return None."""
        tracker = WordCompletionTracker("hello world")
        tracker.add_word_and_check_complete("hello")
        self.assertIsNone(tracker.get_overflow())
        self.assertIsNone(tracker.get_raw_overflow_word())


class TestWordCompletionTrackerMissingWord(unittest.TestCase):
    """Force-complete: TTS provider drops a word-timestamp event."""

    def test_force_complete_when_word_does_not_belong(self):
        """When word doesn't belong, the slot is force-completed and word becomes overflow."""
        tracker = WordCompletionTracker("number is")
        result = tracker.add_word_and_check_complete("4111")
        self.assertTrue(result)
        self.assertEqual(tracker.get_overflow(), "4111")
        self.assertEqual(tracker.get_raw_overflow_word(), "4111")

    def test_force_complete_frame_word_is_full_remaining_expected(self):
        """Force-complete with no prior progress: frame_word is the entire expected text."""
        tracker = WordCompletionTracker("number is")
        tracker.add_word_and_check_complete("4111")
        self.assertEqual(tracker.get_frame_word(), "number is")

    def test_force_complete_frame_word_is_partial_remaining_expected(self):
        """Force-complete after partial progress: frame_word is only the unspoken suffix."""
        tracker = WordCompletionTracker("number is")
        tracker.add_word_and_check_complete("number")  # consumes "number" (6 chars)
        # Now remaining = " is"; "4111" doesn't belong
        result = tracker.add_word_and_check_complete("4111")
        self.assertTrue(result)
        self.assertEqual(tracker.get_frame_word(), "is")
        self.assertEqual(tracker.get_overflow(), "4111")

    def test_force_complete_overflow_routes_to_next_tracker(self):
        """Overflow from a force-completed slot feeds the next tracker correctly."""
        tracker1 = WordCompletionTracker("number is")
        tracker2 = WordCompletionTracker("4111 1111")

        tracker1.add_word_and_check_complete("4111")  # force-completes tracker1
        overflow = tracker1.get_raw_overflow_word()
        self.assertEqual(overflow, "4111")

        self.assertFalse(tracker2.add_word_and_check_complete(overflow))
        self.assertTrue(tracker2.add_word_and_check_complete("1111"))

    def test_force_complete_after_several_normal_words(self):
        """Partial consumption followed by a mismatched word force-completes correctly."""
        tracker = WordCompletionTracker("Your credit card number is")
        for word in ["Your", "credit", "card"]:
            tracker.add_word_and_check_complete(word)
        # "4111" doesn't belong (remaining starts with "number...")
        result = tracker.add_word_and_check_complete("4111")
        self.assertTrue(result)
        # frame_word should be the unspoken suffix including leading space
        self.assertEqual(tracker.get_frame_word(), "number is")
        self.assertEqual(tracker.get_raw_overflow_word(), "4111")

    def test_force_complete_marks_tracker_complete(self):
        """After force-complete, is_complete is True regardless of how many chars were spoken."""
        tracker = WordCompletionTracker("hello world")
        tracker.add_word_and_check_complete("hello")
        self.assertFalse(tracker.is_complete)
        tracker.add_word_and_check_complete("4111")  # force-complete
        self.assertTrue(tracker.is_complete)

    def test_force_complete_does_not_affect_overflow_getters_on_normal_words(self):
        """overflow state is fresh per call; a normal word clears any prior force-complete state."""
        tracker1 = WordCompletionTracker("ab")
        tracker2 = WordCompletionTracker("cd")

        # Force-complete tracker1 with a wrong word
        tracker1.add_word_and_check_complete("xyz")
        self.assertIsNotNone(tracker1.get_overflow())

        # tracker2 receives "cd" normally — no overflow
        tracker2.add_word_and_check_complete("cd")
        self.assertIsNone(tracker2.get_overflow())


class TestWordCompletionTrackerRawText(unittest.TestCase):
    """raw_text cursor tracking: each word maps back to its span in the original text."""

    def test_raw_text_none_when_no_raw_text_provided(self):
        """get_raw_consumed returns None when raw_text was not given."""
        tracker = WordCompletionTracker("hello world")
        tracker.add_word_and_check_complete("hello")
        self.assertIsNone(tracker.get_raw_consumed())

    def test_raw_text_single_word_no_tags(self):
        """Simple case: raw_text matches expected_text, no tags."""
        tracker = WordCompletionTracker("hello world", raw_text="hello world")
        tracker.add_word_and_check_complete("hello")
        self.assertEqual(tracker.get_raw_consumed(), "hello")
        tracker.add_word_and_check_complete("world")
        self.assertEqual(tracker.get_raw_consumed(), "world")

    def test_raw_text_opening_tag_included_in_first_word(self):
        """The opening tag preceding content is consumed with the first word."""
        raw = "<card>4111 1111 1111 1111</card>"
        tracker = WordCompletionTracker("4111 1111 1111 1111", raw_text=raw)
        tracker.add_word_and_check_complete("4111")
        self.assertEqual(tracker.get_raw_consumed(), "<card>4111")

    def test_raw_text_tag_chars_not_counted_as_alnum(self):
        """Tag chars (c,a,r,d) inside <card> must not burn the alnum budget."""
        # If tag chars were counted, "4111" would only consume "<card" (4 alnum budget
        # spent on c,a,r,d) and the result would be wrong.
        raw = "<card>4111</card>"
        tracker = WordCompletionTracker("4111", raw_text=raw)
        tracker.add_word_and_check_complete("4111")
        # The full "<card>4111</card>" should be consumed (last word → consume all).
        self.assertEqual(tracker.get_raw_consumed(), "<card>4111</card>")

    def test_raw_text_four_words_with_card_tags(self):
        """Full card-number scenario: each word maps to its correct raw span."""
        raw = "<card>4111 1111 1111 1111</card>"
        tracker = WordCompletionTracker("4111 1111 1111 1111", raw_text=raw)

        tracker.add_word_and_check_complete("4111")
        self.assertEqual(tracker.get_raw_consumed(), "<card>4111")

        tracker.add_word_and_check_complete("1111")
        self.assertEqual(tracker.get_raw_consumed(), "1111")

        tracker.add_word_and_check_complete("1111")
        self.assertEqual(tracker.get_raw_consumed(), "1111")

        tracker.add_word_and_check_complete("1111")
        # Last word: consume all remaining raw text including closing tag.
        self.assertEqual(tracker.get_raw_consumed(), "1111</card>")
        self.assertTrue(tracker.is_complete)

    def test_raw_text_closing_tag_consumed_with_last_word(self):
        """Closing tag is swept into the last word's raw_consumed, not lost."""
        raw = "<card>hello</card>"
        tracker = WordCompletionTracker("hello", raw_text=raw)
        tracker.add_word_and_check_complete("hello")
        self.assertEqual(tracker.get_raw_consumed(), "<card>hello</card>")

    def test_raw_text_mid_frame_word_does_not_consume_closing_tag(self):
        """Non-final words stop before the closing tag; only the last sweeps it up."""
        raw = "<card>hello world</card>"
        tracker = WordCompletionTracker("hello world", raw_text=raw)
        tracker.add_word_and_check_complete("hello")
        self.assertEqual(tracker.get_raw_consumed(), "<card>hello")
        tracker.add_word_and_check_complete("world")
        self.assertEqual(tracker.get_raw_consumed(), "world</card>")

    def test_raw_text_force_complete_consumes_all_remaining(self):
        """When force-complete fires, all remaining raw_text is consumed at once."""
        raw = "<card>4111 1111 1111 1111</card>"
        tracker = WordCompletionTracker("4111 1111 1111 1111", raw_text=raw)
        tracker.add_word_and_check_complete("4111")  # advances cursor to pos 10
        # "WRONG" doesn't belong → force-complete
        tracker.add_word_and_check_complete("WRONG")
        self.assertEqual(tracker.get_raw_consumed(), "1111 1111 1111</card>")
        self.assertTrue(tracker.is_complete)

    def test_raw_text_overflow_word_last_raw_consumed_sweeps_tag(self):
        """When a straddling word completes the frame, closing tag is consumed with it."""
        raw = "<card>4111 1111</card>"
        tracker = WordCompletionTracker("4111 1111", raw_text=raw)
        tracker.add_word_and_check_complete("4111")
        self.assertEqual(tracker.get_raw_consumed(), "<card>4111")

        # "1111And" straddles into the next frame
        result = tracker.add_word_and_check_complete("1111And")
        self.assertTrue(result)
        self.assertEqual(tracker.get_frame_word(), "1111")
        self.assertEqual(tracker.get_raw_overflow_word(), "And")
        # is_complete → consume all remaining raw including </card>
        self.assertEqual(tracker.get_raw_consumed(), "1111</card>")

    def test_raw_text_with_ssml_tags_in_expected(self):
        """expected_text with SSML tags: raw cursor still advances by content alnum count."""
        # Cartesia might receive "<spell>4111 1111</spell>" as expected_text
        # while raw_text is "<card>4111 1111</card>"
        raw = "<card>4111 1111</card>"
        tracker = WordCompletionTracker("<spell>4111 1111</spell>", raw_text=raw)
        # normalized expected = "41111111" (8 chars); both texts share the same alnum count.
        tracker.add_word_and_check_complete("4111")
        self.assertEqual(tracker.get_raw_consumed(), "<card>4111")
        tracker.add_word_and_check_complete("1111")
        self.assertEqual(tracker.get_raw_consumed(), "1111</card>")
        self.assertTrue(tracker.is_complete)


class TestWordCompletionTrackerMultiFrameSimulation(unittest.TestCase):
    """End-to-end simulations of multiple AggregatedTextFrame slots."""

    def test_two_plain_frames_sequential_words(self):
        """Normal two-frame flow: all words arrive in order with no drops or overflow."""
        tracker1 = WordCompletionTracker("Your credit card")
        tracker2 = WordCompletionTracker("number is 42")

        for word in ["Your", "credit", "card"]:
            tracker1.add_word_and_check_complete(word)
        self.assertTrue(tracker1.is_complete)

        for word in ["number", "is"]:
            self.assertFalse(tracker2.add_word_and_check_complete(word))
        self.assertTrue(tracker2.add_word_and_check_complete("42"))

    def test_credit_card_full_flow_with_raw_text(self):
        """Full card-number scenario with raw_text tracking across two frames."""
        tracker1 = WordCompletionTracker("Your credit card number is")
        tracker2 = WordCompletionTracker(
            "4111 1111 1111 1111",
            raw_text="<card>4111 1111 1111 1111</card>",
        )

        for word in ["Your", "credit", "card", "number", "is"]:
            tracker1.add_word_and_check_complete(word)
        self.assertTrue(tracker1.is_complete)

        tracker2.add_word_and_check_complete("4111")
        self.assertEqual(tracker2.get_raw_consumed(), "<card>4111")

        tracker2.add_word_and_check_complete("1111")
        self.assertEqual(tracker2.get_raw_consumed(), "1111")

        tracker2.add_word_and_check_complete("1111")
        self.assertEqual(tracker2.get_raw_consumed(), "1111")

        result = tracker2.add_word_and_check_complete("1111")
        self.assertTrue(result)
        self.assertEqual(tracker2.get_raw_consumed(), "1111</card>")

    def test_missing_word_force_complete_then_next_frame(self):
        """Dropped word-timestamp: force-complete slot 1, route word to slot 2."""
        tracker1 = WordCompletionTracker("Your credit card number is")
        tracker2 = WordCompletionTracker(
            "4111 1111 1111 1111",
            raw_text="<card>4111 1111 1111 1111</card>",
        )

        # "Your", "credit", "card" arrive; then "number" and "is" are dropped.
        for word in ["Your", "credit", "card"]:
            tracker1.add_word_and_check_complete(word)

        # "4111" arrives but belongs to tracker2 — force-completes tracker1.
        result = tracker1.add_word_and_check_complete("4111")
        self.assertTrue(result)
        # frame_word carries the unspoken remainder so a TTSTextFrame can be emitted.
        self.assertEqual(tracker1.get_frame_word(), "number is")
        overflow = tracker1.get_raw_overflow_word()
        self.assertEqual(overflow, "4111")

        # Route overflow into tracker2.
        tracker2.add_word_and_check_complete(overflow)
        self.assertEqual(tracker2.get_raw_consumed(), "<card>4111")

        tracker2.add_word_and_check_complete("1111")
        tracker2.add_word_and_check_complete("1111")
        result = tracker2.add_word_and_check_complete("1111")
        self.assertTrue(result)
        self.assertEqual(tracker2.get_raw_consumed(), "1111</card>")

    def test_overflow_word_spans_two_frames(self):
        """A word token that straddles two frame boundaries splits and routes correctly."""
        tracker1 = WordCompletionTracker("4111 1111 1111 1111")
        tracker2 = WordCompletionTracker("And your")

        for word in ["4111", "1111", "1111"]:
            tracker1.add_word_and_check_complete(word)

        # "1111And" spans the frame boundary.
        result = tracker1.add_word_and_check_complete("1111And")
        self.assertTrue(result)
        self.assertEqual(tracker1.get_frame_word(), "1111")
        self.assertEqual(tracker1.get_overflow(), "and")
        self.assertEqual(tracker1.get_raw_overflow_word(), "And")

        # Feed overflow into tracker2.
        result = tracker2.add_word_and_check_complete(tracker1.get_raw_overflow_word())
        self.assertFalse(result)
        self.assertEqual(tracker2.get_frame_word(), "And")

        self.assertTrue(tracker2.add_word_and_check_complete("your"))

    def test_overflow_with_raw_text_across_frames(self):
        """Raw text cursor is correct when the last straddling word completes the frame."""
        tracker1 = WordCompletionTracker(
            "4111 1111",
            raw_text="<card>4111 1111</card>",
        )
        tracker2 = WordCompletionTracker("And")

        tracker1.add_word_and_check_complete("4111")
        self.assertEqual(tracker1.get_raw_consumed(), "<card>4111")

        # "1111And" completes tracker1; "And" overflows to tracker2.
        result = tracker1.add_word_and_check_complete("1111And")
        self.assertTrue(result)
        self.assertEqual(tracker1.get_raw_consumed(), "1111</card>")
        self.assertEqual(tracker1.get_raw_overflow_word(), "And")

        result = tracker2.add_word_and_check_complete(tracker1.get_raw_overflow_word())
        self.assertTrue(result)
        self.assertEqual(tracker2.get_frame_word(), "And")

    def test_multiple_missing_words_single_force_complete(self):
        """Even if several consecutive words are dropped, one force-complete handles them all."""
        # Frame expects "one two three"; TTS skips straight to "four" (next frame's word).
        tracker1 = WordCompletionTracker("one two three")
        tracker2 = WordCompletionTracker("four five")

        # No words from tracker1 ever arrive; "four" is the first word seen.
        result = tracker1.add_word_and_check_complete("four")
        self.assertTrue(result)
        self.assertEqual(tracker1.get_frame_word(), "one two three")
        self.assertEqual(tracker1.get_raw_overflow_word(), "four")

        self.assertFalse(tracker2.add_word_and_check_complete("four"))
        self.assertTrue(tracker2.add_word_and_check_complete("five"))


if __name__ == "__main__":
    unittest.main()
