#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Word completion tracker for TTS context ordering."""

import re
from loguru import logger


class WordCompletionTracker:
    """Tracks whether all words from a source AggregatedTextFrame have been spoken.

    Compares normalized alphanumeric character counts between the expected text
    and accumulated spoken words, making the check robust to punctuation, spacing,
    and XML/HTML tags (e.g. SSML tags like ``<spell>...</spell>`` returned by some
    TTS providers in word-timestamp events).

    Example::

        tracker = WordCompletionTracker("Hello, world!")
        tracker.add_word_and_check_complete("Hello")   # False
        tracker.add_word_and_check_complete("world")   # True  — normalized "helloworld" >= "helloworld"
    """

    def __init__(self, expected_text: str):
        """Initialize the tracker with the text of the frame being spoken.

        Args:
            expected_text: Full text of the AggregatedTextFrame sent to TTS.
        """
        self._expected = self._normalize(expected_text)
        self._received = ""
        logger.info(f"WordCompletionTracker: {self._expected}")

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip XML/HTML tags then keep only lowercase alphanumeric characters."""
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def add_word_and_check_complete(self, word: str) -> bool:
        """Record a spoken word from a word-timestamp event.

        Args:
            word: A single word token returned by the TTS service.

        Returns:
            True when all expected content has been covered.
        """
        logger.info(f"WordCompletionTracker add_word_and_check_complete: {word}")
        self._received += self._normalize(word)
        return self.is_complete

    @property
    def is_complete(self) -> bool:
        """True when accumulated normalized chars >= expected normalized chars."""
        return len(self._received) >= len(self._expected)

    def reset(self):
        """Reset received word accumulation without changing the expected text."""
        self._received = ""
