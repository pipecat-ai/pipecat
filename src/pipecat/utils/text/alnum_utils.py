#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared alphanumeric utilities for text normalization and cursor advancement."""

import unicodedata

from pipecat.utils.text.markup_utils import strip_complete_markup


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


_TYPOGRAPHIC_FOLD = str.maketrans(
    {
        "‘": "'",  # ‘ LEFT SINGLE QUOTATION MARK
        "’": "'",  # ’ RIGHT SINGLE QUOTATION MARK
        "ʼ": "'",  # ʼ MODIFIER LETTER APOSTROPHE
        "“": '"',  # “ LEFT DOUBLE QUOTATION MARK
        "”": '"',  # ” RIGHT DOUBLE QUOTATION MARK
        "–": "-",  # – EN DASH
        "—": "-",  # — EM DASH
    }
)
"""Typographic punctuation variants mapped to their ASCII equivalents.

LLMs emit the typographic forms and TTS services may report the ASCII ones in
word-timestamp events (or the reverse). Every entry is a single character mapped
to a single character, which is what lets :func:`fold_for_matching` keep its
1:1 length contract.
"""


def fold_typography(text: str) -> str:
    """Replace typographic punctuation variants with their ASCII equivalents.

    Args:
        text: Input text to fold.

    Returns:
        *text* with typographic quotes and dashes replaced by ASCII; same
        length as *text*.
    """
    return text.translate(_TYPOGRAPHIC_FOLD)


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


def fold_for_matching(text: str) -> str:
    """Fold away surface variation between two spellings of the same text, 1:1.

    Unlike :func:`alnum_only`, this never removes or merges characters --
    punctuation, spaces, and markup are passed through unchanged, and each
    output character corresponds to exactly the same-index input character. A
    raw offset computed against the folded text therefore applies unchanged to
    the original, so callers can use it as a drop-in transform before a
    position-based literal comparison, without the risk of a fully-normalized
    (whitespace/punctuation-stripped) comparison matching across a boundary
    that wasn't already a candidate in the untransformed comparison.

    Folds case, accents, and typographic punctuation (``\u2019`` -> ``'``, ``\u2013`` -> ``-``) --
    the variations a TTS service may introduce between the text it was sent and the
    words it reports back. Deliberately narrow: each folded character is listed in
    :data:`_TYPOGRAPHIC_FOLD`, rather than applying a blanket Unicode compatibility
    normalization, which would silently fold thousands of characters (CJK compatibility
    ideographs, halfwidth katakana, math alphanumerics) that no service is known to
    substitute.

    Args:
        text: Input text to fold.

    Returns:
        *text* with those variations folded away; same length as *text*.
    """
    folded = "".join(_fold_accented_char(ch) if ch.isalpha() else ch for ch in text)
    return fold_typography(folded)


def alnum_only(text: str) -> str:
    """Strip XML/HTML tags then keep only lowercase alphanumeric characters.

    Accented letters (e.g. ã, é) are reduced to their base letter so TTS output
    can be matched against LLM text even when the provider strips diacritics.
    Non-Latin scripts (CJK, Hangul) are kept as-is — each original character
    contributes exactly one char to the result, keeping normalized length in sync
    with raw alnum counts used by advance_by_alnums.

    Args:
        text: Input text to reduce.

    Returns:
        Lowercase alphanumeric-only string with tags stripped.
    """
    text = strip_complete_markup(text)
    result = []
    for char in text:
        # Ignore punctuation, spaces, emojis, etc.
        # Keep only letters and numbers.
        if not char.isalnum():
            continue
        if char.isalpha():
            # Letters, including CJK and Hangul (both alphabetic per
            # str.isalpha()): fold accents, a no-op for scripts that have none.
            result.append(_fold_accented_char(char))
        else:
            # Digits and other alnum-but-not-alphabetic characters: no case or
            # accent to fold, so keep as-is (lowercase conversion is a no-op).
            result.append(char.lower())
    return "".join(result)


def has_alnum(text: str) -> bool:
    """Return True if *text* holds anything alphanumeric once markup is stripped.

    The predicate form of :func:`alnum_only`, for the question the callers
    actually ask: is there anything left to speak here? A tag's letters do not
    count -- ``"<break/>"`` is empty by this measure.
    """
    return bool(alnum_only(text))


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
