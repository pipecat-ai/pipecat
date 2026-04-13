#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Inworld TTS punctuation restoration logic."""

import unittest

from pipecat.services.inworld.tts import _restore_punctuation


class TestRestorePunctuation(unittest.TestCase):
    def test_basic_punctuation_restoration(self):
        """Bare aligned words are mapped back to punctuated originals."""
        aligned = ["Hello", "how", "are", "you"]
        original = ["Hello,", "how", "are", "you?"]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["Hello,", "how", "are", "you?"])
        self.assertEqual(idx, 4)

    def test_partial_punctuation_from_api(self):
        """When the API includes some punctuation, the original token is still preferred."""
        aligned = ["Hello,", "world"]
        original = ["Hello,", "world!"]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["Hello,", "world!"])
        self.assertEqual(idx, 2)

    def test_contractions(self):
        """Contractions with straight apostrophes match correctly."""
        aligned = ["don't", "worry"]
        original = ["don't,", "worry."]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["don't,", "worry."])
        self.assertEqual(idx, 2)

    def test_curly_apostrophe(self):
        """Contractions with curly apostrophes (U+2019) match straight ones."""
        aligned = ["don\u2019t"]
        original = ["don\u2019t,"]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["don\u2019t,"])
        self.assertEqual(idx, 1)

    def test_cross_apostrophe_matching(self):
        """Straight apostrophe in aligned matches curly apostrophe in original."""
        aligned = ["don't"]
        original = ["don\u2019t,"]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["don\u2019t,"])
        self.assertEqual(idx, 1)

    def test_number_expansion_fallback(self):
        """TTS-expanded numbers fall back to bare aligned words."""
        aligned = ["one", "hundred"]
        original = ["100"]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["one", "hundred"])
        self.assertEqual(idx, 0)

    def test_hyphenated_word_fallback(self):
        """Hyphenated words split by TTS fall back gracefully."""
        aligned = ["mother", "in", "law"]
        original = ["mother-in-law"]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["mother", "in", "law"])
        self.assertEqual(idx, 0)

    def test_repeated_words(self):
        """Repeated words are matched sequentially via cursor advancement."""
        aligned = ["the", "cat", "and", "the", "dog"]
        original = ["the", "cat", "and", "the", "dog."]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["the", "cat", "and", "the", "dog."])
        self.assertEqual(idx, 5)

    def test_empty_aligned_words(self):
        """Empty aligned words returns empty list."""
        result, idx = _restore_punctuation([], ["Hello,", "world!"])
        self.assertEqual(result, [])
        self.assertEqual(idx, 0)

    def test_empty_original_tokens(self):
        """Empty original tokens causes all words to fall back."""
        aligned = ["Hello", "world"]
        result, idx = _restore_punctuation(aligned, [])
        self.assertEqual(result, ["Hello", "world"])
        self.assertEqual(idx, 0)

    def test_cursor_continuity_across_calls(self):
        """Simulates multiple chunks within one context — cursor resumes."""
        original = ["Hello,", "how", "are", "you?", "I'm", "fine."]

        # First chunk
        result1, idx = _restore_punctuation(["Hello", "how"], original, start_idx=0)
        self.assertEqual(result1, ["Hello,", "how"])
        self.assertEqual(idx, 2)

        # Second chunk — cursor resumes from idx=2
        result2, idx = _restore_punctuation(["are", "you"], original, start_idx=idx)
        self.assertEqual(result2, ["are", "you?"])
        self.assertEqual(idx, 4)

    def test_case_insensitive_matching(self):
        """Matching is case-insensitive to handle capitalization differences."""
        aligned = ["hello"]
        original = ["Hello,"]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["Hello,"])
        self.assertEqual(idx, 1)

    def test_punctuation_only_text(self):
        """Aligned word of pure punctuation doesn't match anything, falls back."""
        aligned = ["..."]
        original = ["Hello,"]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(result, ["..."])
        self.assertEqual(idx, 0)

    def test_multi_sentence(self):
        """Full multi-sentence restoration works end-to-end."""
        aligned = ["Hey", "welcome", "back", "It's", "good", "to", "have", "you", "again"]
        original = ["Hey,", "welcome", "back!", "It's", "good", "to", "have", "you", "again."]
        result, idx = _restore_punctuation(aligned, original)
        self.assertEqual(
            result,
            ["Hey,", "welcome", "back!", "It's", "good", "to", "have", "you", "again."],
        )
        self.assertEqual(idx, 9)


if __name__ == "__main__":
    unittest.main()
