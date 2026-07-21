#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared alphanumeric utilities for text normalization and cursor advancement."""

import re
import unicodedata


def strip_trailing_punctuation(text: str) -> str:
    """Remove punctuation only at the very end of *text*.

    Args:
        text: Input text to trim.

    Returns:
        *text* with any trailing run of Unicode punctuation characters removed.
    """
    i = len(text)
    while i > 0 and unicodedata.category(text[i - 1]).startswith("P"):
        i -= 1
    return text[:i]


def _fold_accented_char(char: str) -> str:
    """Lowercase *char*, reduced to its base letter if it carries a combining accent.

    NFD decomposes an accented character into a base letter plus a combining
    mark (e.g. ``é`` -> ``e`` + ``◌́``, category ``Mn``); dropping the mark
    keeps only the base letter. Always returns exactly one character, so
    callers can rely on a 1:1 length mapping with the input.
    """
    nfd = unicodedata.normalize("NFD", char)
    if len(nfd) >= 2 and unicodedata.category(nfd[1]) == "Mn":
        return nfd[0].lower()
    return char.lower()


def fold_case_and_accents(text: str) -> str:
    """Lowercase letters and strip accents, preserving every other character 1:1.

    Unlike :func:`normalize`, this never removes or merges characters --
    punctuation, spaces, and markup are passed through unchanged, and each
    output character corresponds to exactly the same-index input character. A
    raw offset computed against the folded text therefore applies unchanged to
    the original, so callers can use it as a drop-in transform before a
    position-based literal comparison, without the risk of a fully-normalized
    (whitespace/punctuation-stripped) comparison matching across a boundary
    that wasn't already a candidate in the untransformed comparison.

    Args:
        text: Input text to fold.

    Returns:
        *text* with letters case- and accent-folded; same length as *text*.
    """
    return "".join(_fold_accented_char(ch) if ch.isalpha() else ch for ch in text)


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
        if char.isalpha():
            result.append(_fold_accented_char(char))
        else:
            # Regular numbers, CJK, Hangul, etc. are kept unchanged (except
            # lowercase conversion, a no-op for characters with no case).
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
