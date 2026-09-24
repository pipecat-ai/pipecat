#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utilities for normalizing word-timestamp streams from TTS services."""

import re

# A word ending in a digit, optionally followed by a decimal point or a thousands
# separator, may still be continued by the digits that follow it.
_NUMBER_MAY_CONTINUE_RE = re.compile(r"\d[.,]?$")


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _may_continue_number(group: list[tuple[str, float]]) -> bool:
    return bool(_NUMBER_MAY_CONTINUE_RE.search(_strip_tags("".join(w for w, _ in group))))


def _group_word_tokens(
    word_times: list[tuple[str, float]],
) -> list[list[tuple[str, float]]]:
    """Group raw tokens into the words they belong to.

    A token with no alphanumeric content (after stripping XML/HTML tags) joins the
    preceding word. A token starting with a digit joins a preceding word that ends in
    a digit, optionally followed by ``.`` or ``,``: some TTS services (e.g. Inworld)
    report the digits of one number as separate tokens with no space between them
    (``"2"``, ``"5"``, ``"0"`` for ``"250"``). Punct/space tokens with no preceding
    word are dropped.
    """
    groups: list[list[tuple[str, float]]] = []
    for word, ts in word_times:
        stripped = _strip_tags(word)
        has_alnum = any(c.isalnum() for c in stripped)
        if groups and (
            not has_alnum or (stripped[:1].isdigit() and _may_continue_number(groups[-1]))
        ):
            groups[-1].append((word, ts))
        elif has_alnum:
            groups.append([(word, ts)])
    return groups


def merge_punct_tokens(
    word_times: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Merge punctuation/space-only tokens and number fragments into their word.

    Some TTS services (e.g. Inworld) emit spaces and punctuation as separate
    word-timestamp tokens rather than attaching them to the adjacent word, and report
    the digits of one number as separate tokens. This function collapses those tokens
    so downstream consumers always receive whole words with trailing punctuation
    already attached — identical to the format produced by ElevenLabs or Cartesia.

    A token is considered punct/space-only when its text contains no alphanumeric
    characters after stripping XML/HTML tags.  Such tokens are appended to the
    preceding word's text and their timestamp is discarded (the preceding word's
    timestamp is kept).  A token starting with a digit is appended the same way to a
    preceding word that ends in a digit, optionally followed by ``.`` or ``,``, so
    ``"3"``, ``","``, ``"5"``, ``"0"``, ``"0"`` becomes ``"3,500"``.  Leading
    punct/space tokens with no preceding word are silently discarded.  Every output
    token is stripped of leading and trailing whitespace (spaces, tabs, newlines).

    Args:
        word_times: Raw list of ``(word, timestamp)`` pairs from the TTS service.

    Returns:
        Merged list where every entry contains at least one alphanumeric character
        and has no leading or trailing whitespace.

    Example::

        merge_punct_tokens([("questions", 1.0), (", ", 1.2), ("explain", 1.4)])
        # → [("questions,", 1.0), ("explain", 1.4)]
        merge_punct_tokens([("2", 1.0), ("5", 1.1), ("0", 1.2), (" ", 1.3)])
        # → [("250", 1.0)]
    """
    return [
        ("".join(word for word, _ in group).strip(), group[0][1])
        for group in _group_word_tokens(word_times)
    ]


def split_trailing_number(
    word_times: list[tuple[str, float]],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Split off the raw tokens of a trailing number that later tokens may continue.

    A service that delivers word timestamps in several messages can end a message in
    the middle of a number (``"20"`` in one message, ``"26"`` in the next). The caller
    holds the trailing tokens and prepends them to the next message's tokens, so
    :func:`merge_punct_tokens` sees the whole number. Only a last word ending in a
    digit, optionally followed by ``.`` or ``,``, is held; any other word is complete.

    Args:
        word_times: Raw list of ``(word, timestamp)`` pairs from the TTS service.

    Returns:
        ``(complete, trailing)``: the raw tokens that can be merged now, and the raw
        tokens of a trailing number (empty when there is none).
    """
    groups = _group_word_tokens(word_times)
    if groups and _may_continue_number(groups[-1]):
        return [token for group in groups[:-1] for token in group], groups[-1]
    return word_times, []
