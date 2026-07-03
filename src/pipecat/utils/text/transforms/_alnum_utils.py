#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared alphanumeric utilities for text normalization and cursor advancement."""

import re
import unicodedata


def normalize(text: str) -> str:
    """Strip XML/HTML tags then keep only lowercase alphanumeric characters.

    Accented letters (e.g. ã, é) are reduced to their base letter so TTS output
    can be matched against LLM text even when the provider strips diacritics.
    Non-Latin scripts (CJK, Hangul) are kept as-is — each original character
    contributes exactly one char to the result, keeping normalized length in sync
    with raw alnum counts used by advance_by_alnums.

    Args:
        text: Input text to normalize.

    Returns:
        Lowercase alphanumeric-only string with tags stripped.
    """
    text = re.sub(r"<[^>]+>", "", text)
    result = []
    for char in text:
        # Ignore punctuation, spaces, emojis, etc.
        # Keep only letters and numbers.
        if not char.isalnum():
            continue
        # NFD decomposes accented characters into:
        #   é -> e + ◌́
        #   ã -> a + ◌̃
        #
        # Non-accented characters usually stay unchanged.
        nfd = unicodedata.normalize("NFD", char)
        # Unicode category "Mn" means:
        #   Mark, Nonspacing
        #
        # These are combining accent marks that modify
        # the previous character but are not standalone.
        #
        # Example:
        #   "é" becomes:
        #       nfd[0] = "e"
        #       nfd[1] = "◌́"  (category = "Mn")
        #
        # If the second character is a combining accent,
        # keep only the base letter.
        if len(nfd) >= 2 and unicodedata.category(nfd[1]) == "Mn":
            # Accented letter: keep the base character only (drops the combining mark).
            result.append(nfd[0].lower())
        else:
            # Regular ASCII, numbers, CJK, Hangul, etc.
            # are kept unchanged (except lowercase conversion).
            result.append(char.lower())
    return "".join(result)


def advance_by_alnums(text: str, start_pos: int, n: int) -> int:
    """Return the position in *text* after advancing past *n* alphanumeric chars.

    Moves through the text one character at a time, counting only alphanumeric
    characters. XML/HTML tags (``<...>``) are skipped entirely — their content
    is not counted against the budget, so the returned span includes the full tag.
    Other non-alphanumeric characters (spaces, punctuation) are also passed over
    without decrementing the budget.

    After the *n* alnum chars are consumed, advances further past any immediately
    following punctuation (e.g. the ``,`` in ``"questions,"`` or the ``.`` in
    ``"done."``), stopping before the next space, alnum char, or XML tag.

    Args:
        text: The source text to scan.
        start_pos: Starting position in *text*.
        n: Number of alphanumeric characters to consume.

    Returns:
        New position in *text* after consuming *n* alnum chars and trailing punctuation.
    """
    pos = start_pos
    count = 0
    while pos < len(text) and count < n:
        if text[pos] == "<":
            end = text.find(">", pos)
            pos = end + 1 if end != -1 else pos + 1
        elif text[pos].isalnum():
            count += 1
            pos += 1
        else:
            pos += 1

    while pos < len(text):
        if text[pos] == "<":
            break
        if text[pos].isalnum() or text[pos].isspace():
            break
        pos += 1

    return pos
