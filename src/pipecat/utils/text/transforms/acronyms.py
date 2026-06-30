#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Normalize acronyms so TTS pronounces each letter individually."""

import re

from pipecat.frames.frames import AggregationType

# Two or more consecutive uppercase letters not immediately followed by a lowercase
# letter (to avoid splitting CamelCase words like "iPhone" or "McDonalds").
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?![a-z])\b")


async def normalize_acronyms(text: str, aggregation_type: str | AggregationType) -> str:
    """Insert spaces between letters of uppercase acronyms.

    This transformer is alphanumeric-preserving: the same letters are kept,
    only spaces are added between them, so the :class:`WordCompletionTracker`
    requires no segment-map overhead.

    Args:
        text: Input text possibly containing acronyms like API or HTTP.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with acronyms letter-spaced (e.g. ``"API"`` → ``"A P I"``).

    Example::

        result = await normalize_acronyms("Use the API or HTTP endpoint", "*")
        # "Use the A P I or H T T P endpoint"
    """
    return _ACRONYM_RE.sub(lambda m: " ".join(m.group(0)), text)
