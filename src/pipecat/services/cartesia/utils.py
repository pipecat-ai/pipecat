#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utilities for Cartesia service implementations."""


def _is_cjk_language(language: str) -> bool:
    base_lang = language.split("-")[0].lower()
    return base_lang in {"zh", "ja", "ko"}


def _is_korean_language(language: str) -> bool:
    base_lang = language.split("-")[0].lower()
    return base_lang == "ko"


def process_word_timestamps_for_language(
    words: list[str],
    starts: list[float],
    language: str | None,
) -> list[tuple[str, float]]:
    """Normalize Cartesia word timestamps for the current language.

    Args:
        words: Words or characters from Cartesia.
        starts: Start timestamps for each item in ``words``.
        language: Language code to normalize for.

    Returns:
        Word timestamp pairs normalized for downstream transcript aggregation.
    """
    if language and _is_cjk_language(language):
        if words and starts:
            separator = " " if _is_korean_language(language) else ""
            combined_word = separator.join(words)
            return [(combined_word, starts[0])]
        else:
            return []
    else:
        return list(zip(words, starts))
