#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utilities for Cartesia service implementations."""


def is_cjk_language(language: str) -> bool:
    """Check if the given language is CJK (Chinese, Japanese, Korean).

    Args:
        language: The language code to check.

    Returns:
        True if the language is Chinese, Japanese, or Korean.
    """
    cjk_languages = {"zh", "ja", "ko"}
    base_lang = language.split("-")[0].lower()
    return base_lang in cjk_languages


def is_korean_language(language: str) -> bool:
    """Check if the given language is Korean.

    Args:
        language: The language code to check.

    Returns:
        True if the language is Korean.
    """
    base_lang = language.split("-")[0].lower()
    return base_lang == "ko"


def process_word_timestamps_for_language(
    words: list[str],
    starts: list[float],
    language: str | None,
) -> list[tuple[str, float]]:
    """Process Cartesia word timestamps based on the current language.

    For CJK languages, Cartesia groups related characters in the same timestamp message.
    For example, in Japanese a single message might be `['こ', 'ん', 'に', 'ち', 'は', '。']`.
    We combine these into single timestamp entries so the downstream aggregator can add
    natural spacing between meaningful units rather than individual characters. Chinese
    and Japanese do not use inter-word spaces, but Korean does.

    For non-CJK languages, words are already properly separated and are used as-is.

    Args:
        words: List of words/characters from Cartesia.
        starts: List of start timestamps for each word/character.
        language: Language code to process timestamps for.

    Returns:
        List of (word, start_time) tuples processed for the language.
    """
    if language and is_cjk_language(language):
        # For CJK languages, combine all characters in this message into one timestamp
        # entry using the first character's start time. Korean uses spaces between
        # words, while Chinese and Japanese do not.
        if words and starts:
            separator = " " if is_korean_language(language) else ""
            combined_word = separator.join(words)
            first_start = starts[0]
            return [(combined_word, first_start)]
        else:
            return []
    else:
        return list(zip(words, starts))
