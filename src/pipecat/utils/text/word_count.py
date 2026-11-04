#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Spoken-word counting for transcripts that may omit spaces."""

import unicodedata

# Everyday Chinese and Japanese letters. Hangul is omitted: Korean separates
# words with spaces, so a Hangul token stays one word.
_CJK_RANGES = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0xFF66, 0xFF9D),  # Halfwidth Katakana
)


def _is_cjk_word_char(char: str) -> bool:
    """Whether *char* is a Chinese or Japanese letter, not punctuation in those blocks."""
    code = ord(char)
    if not any(start <= code <= end for start, end in _CJK_RANGES):
        return False
    return not unicodedata.category(char).startswith("P")


def count_spoken_words(text: str) -> int:
    """Count spoken words, treating each Chinese or Japanese character as a word.

    Whitespace-separated tokens count as one word, the same as ``str.split``.
    Chinese and Japanese are written without spaces between words, so each
    ideograph, hiragana, or katakana character inside a token counts as one
    word. Punctuation in that token does not. A run of other characters in the
    same token, such as a Latin word or a number, still counts as one word.

    Args:
        text: Transcript or reply text.

    Returns:
        The number of spoken words.
    """
    total = 0
    for token in text.split():
        if not any(_is_cjk_word_char(char) for char in token):
            total += 1
            continue
        in_run = False
        for char in token:
            if _is_cjk_word_char(char):
                total += 1
                in_run = False
            elif unicodedata.category(char).startswith("P"):
                in_run = False
            elif not in_run:
                total += 1
                in_run = True
    return total
