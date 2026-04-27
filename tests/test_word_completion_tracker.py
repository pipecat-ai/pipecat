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
        tracker.add_word("Hello")
        self.assertFalse(tracker.is_complete)
        tracker.add_word("world")
        self.assertTrue(tracker.is_complete)

    def test_add_word_return_value(self):
        """add_word returns True exactly when the tracker becomes complete."""
        tracker = WordCompletionTracker("Hi there")
        self.assertFalse(tracker.add_word("Hi"))
        self.assertTrue(tracker.add_word("there"))

    def test_single_word(self):
        tracker = WordCompletionTracker("Hello")
        self.assertTrue(tracker.add_word("Hello"))

    def test_complete_stays_true_after_extra_words(self):
        """Additional words beyond the expected text keep the tracker complete."""
        tracker = WordCompletionTracker("Hi")
        tracker.add_word("Hi")
        tracker.add_word("extra")
        self.assertTrue(tracker.is_complete)


class TestWordCompletionTrackerNormalization(unittest.TestCase):
    def test_punctuation_ignored_in_expected(self):
        """Punctuation in the source text is stripped before comparison."""
        tracker = WordCompletionTracker("Hello, world!")
        tracker.add_word("Hello")
        tracker.add_word("world")
        self.assertTrue(tracker.is_complete)

    def test_punctuation_ignored_in_words(self):
        """Punctuation attached to TTS word tokens is also stripped."""
        tracker = WordCompletionTracker("Hello world")
        tracker.add_word("Hello,")
        tracker.add_word("world!")
        self.assertTrue(tracker.is_complete)

    def test_case_insensitive(self):
        tracker = WordCompletionTracker("HELLO WORLD")
        tracker.add_word("hello")
        tracker.add_word("world")
        self.assertTrue(tracker.is_complete)

    def test_spaces_ignored(self):
        """Spaces in expected or received text do not count towards char totals."""
        tracker = WordCompletionTracker("a b c")
        # Normalized expected: "abc" (3 chars)
        tracker.add_word("a")
        tracker.add_word("b")
        self.assertFalse(tracker.is_complete)
        tracker.add_word("c")
        self.assertTrue(tracker.is_complete)

    def test_numbers_kept(self):
        """Digits are treated as regular alphanumeric characters."""
        tracker = WordCompletionTracker("Room 42")
        tracker.add_word("Room")
        self.assertFalse(tracker.is_complete)
        tracker.add_word("42")
        self.assertTrue(tracker.is_complete)

    def test_mixed_punctuation_and_numbers(self):
        tracker = WordCompletionTracker("It costs $9.99!")
        # Normalized expected: "itcosts999"
        tracker.add_word("It")
        tracker.add_word("costs")
        self.assertFalse(tracker.is_complete)
        tracker.add_word("$9.99!")
        self.assertTrue(tracker.is_complete)

    def test_special_characters_only_in_expected(self):
        """If the expected text normalizes to empty, the tracker is immediately complete."""
        tracker = WordCompletionTracker("...")
        self.assertTrue(tracker.is_complete)

    def test_special_characters_only_in_word(self):
        """A word token that normalizes to empty contributes nothing."""
        tracker = WordCompletionTracker("hello")
        tracker.add_word("---")  # normalizes to ""
        self.assertFalse(tracker.is_complete)
        tracker.add_word("hello")
        self.assertTrue(tracker.is_complete)


class TestWordCompletionTrackerReset(unittest.TestCase):
    def test_reset_clears_progress(self):
        tracker = WordCompletionTracker("Hello world")
        tracker.add_word("Hello")
        tracker.add_word("world")
        self.assertTrue(tracker.is_complete)
        tracker.reset()
        self.assertFalse(tracker.is_complete)

    def test_reset_allows_reuse(self):
        tracker = WordCompletionTracker("Hello world")
        tracker.add_word("Hello")
        tracker.add_word("world")
        tracker.reset()
        tracker.add_word("Hello")
        self.assertFalse(tracker.is_complete)
        tracker.add_word("world")
        self.assertTrue(tracker.is_complete)

    def test_reset_preserves_expected_text(self):
        """reset() only clears received chars; the expected text is unchanged."""
        tracker = WordCompletionTracker("Hi")
        tracker.add_word("Hi")
        tracker.reset()
        # Re-adding the same word should complete again
        self.assertTrue(tracker.add_word("Hi"))


class TestWordCompletionTrackerEdgeCases(unittest.TestCase):
    def test_empty_expected_text(self):
        """An empty expected string is complete from the start."""
        tracker = WordCompletionTracker("")
        self.assertTrue(tracker.is_complete)

    def test_empty_word_adds_nothing(self):
        tracker = WordCompletionTracker("hello")
        tracker.add_word("")
        self.assertFalse(tracker.is_complete)

    def test_partial_word_completion(self):
        """Chars accumulate; completion happens mid-add_word if count is reached."""
        tracker = WordCompletionTracker("ab")
        # "ab" normalizes to 2 chars; one word covering both at once
        self.assertTrue(tracker.add_word("ab"))

    def test_word_with_extra_chars_completes(self):
        """A single verbose token can satisfy a longer expected text."""
        tracker = WordCompletionTracker("Hi")
        self.assertTrue(tracker.add_word("Hieveryone"))


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
        results = [tracker.add_word(w) for w in self.TTS_WORDS]
        self.assertTrue(results[-1], "tracker should be complete after the last word")

    def test_not_complete_before_last_word(self):
        """Tracker must not report complete before the final word is added."""
        tracker = WordCompletionTracker(self.SENTENCE)
        for word in self.TTS_WORDS[:-1]:
            self.assertFalse(tracker.is_complete, f"should not be complete after '{word}'")

    def test_last_word_triggers_completion(self):
        """Only the last add_word call should return True."""
        tracker = WordCompletionTracker(self.SENTENCE)
        intermediate = [tracker.add_word(w) for w in self.TTS_WORDS[:-1]]
        self.assertFalse(any(intermediate))
        self.assertTrue(tracker.add_word(self.TTS_WORDS[-1]))

    def test_emoji_in_expected_does_not_block_completion(self):
        """The 😊 at the end normalizes to '' so it adds no required chars."""
        tracker = WordCompletionTracker(self.SENTENCE)
        for word in self.TTS_WORDS:
            tracker.add_word(word)
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
            tracker.add_word(chunk)
        self.assertTrue(tracker.is_complete)

    def test_reset_and_replay(self):
        """After reset the same tracker can complete the same sentence again."""
        tracker = WordCompletionTracker(self.SENTENCE)
        for word in self.TTS_WORDS:
            tracker.add_word(word)
        self.assertTrue(tracker.is_complete)

        tracker.reset()
        self.assertFalse(tracker.is_complete)

        for word in self.TTS_WORDS:
            tracker.add_word(word)
        self.assertTrue(tracker.is_complete)

    def test_mid_sentence_is_not_complete(self):
        """Spot-check several points through the sentence."""
        tracker = WordCompletionTracker(self.SENTENCE)
        checkpoints = {4, 9, 14}  # after 5th, 10th, 15th word (0-indexed)
        for i, word in enumerate(self.TTS_WORDS[:-1]):
            tracker.add_word(word)
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
            self.assertFalse(tracker.add_word(word))
        self.assertTrue(tracker.add_word(words[-1]))

    def test_sentence_with_numbers(self):
        """Numbers are alphanumeric and must be counted in both directions."""
        sentence = "There are 3 options available for you."
        words = ["There", "are", "3", "options", "available", "for", "you."]
        tracker = WordCompletionTracker(sentence)
        for word in words[:-1]:
            self.assertFalse(tracker.add_word(word))
        self.assertTrue(tracker.add_word(words[-1]))


if __name__ == "__main__":
    unittest.main()
