#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utilities for normalizing word-timestamp streams from TTS services."""

import re
from collections.abc import Sequence

from pipecat.utils.context.word_timestamp_tracker import WordTime, word_time_end


def merge_punct_tokens(word_times: Sequence[WordTime]) -> list[WordTime]:
    """Merge punctuation/space-only tokens into the preceding word.

    Some TTS services (e.g. Inworld) emit spaces and punctuation as separate
    word-timestamp tokens rather than attaching them to the adjacent word.
    This function collapses those tokens so downstream consumers always receive
    words with trailing punctuation already attached — identical to the format
    produced by ElevenLabs or Cartesia.

    A token is considered punct/space-only when its text contains no alphanumeric
    characters after stripping XML/HTML tags.  Such tokens are appended to the
    preceding word's text and their timestamp is discarded (the preceding word's
    timestamp is kept); when tokens carry an end, the merged word ends where the
    last token merged into it does.  Leading punct/space tokens with no preceding word are
    silently discarded.  Every output token is stripped of leading and trailing
    whitespace (spaces, tabs, newlines).

    Args:
        word_times: Raw list of ``(word, timestamp)`` pairs, or ``(word, timestamp,
            end)`` triples, from the TTS service.

    Returns:
        Merged list where every entry contains at least one alphanumeric character
        and has no leading or trailing whitespace.

    Example::

        merge_punct_tokens([("questions", 1.0), (", ", 1.2), ("explain", 1.4)])
        # → [("questions,", 1.0), ("explain", 1.4)]
    """
    merged: list[list] = []
    for word_time in word_times:
        word = word_time[0]
        stripped = re.sub(r"<[^>]+>", "", word)
        has_alnum = any(c.isalnum() for c in stripped)
        if not has_alnum:
            if merged:
                merged[-1][0] += word
                end = word_time_end(word_time)
                if end is not None:
                    merged[-1][2:] = [end]
            # else: leading punct/space with no preceding word → discard
        else:
            merged.append(list(word_time))
    return [(word.strip(), *rest) for word, *rest in merged]
