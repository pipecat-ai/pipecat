#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand percentage expressions into spoken form for TTS."""

import re

from pipecat.frames.frames import AggregationType

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _percent_to_words(match: re.Match) -> str:
    value = match.group(1)
    try:
        from num2words import num2words

        number_str = num2words(float(value), lang="en")
    except ImportError:
        number_str = value
    return f"{number_str} percent"


async def expand_percentages(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand percentage expressions to their spoken form.

    Requires the ``num2words`` package for numeric word conversion
    (``pip install pipecat-ai[voice-formatting]``). Falls back to keeping the
    numeric digits when the package is not installed.

    Args:
        text: Input text possibly containing percentage expressions.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with percentages replaced by spoken equivalents.

    Example::

        result = await expand_percentages("50% off today", "*")
        # "fifty percent off today"
    """
    return _PERCENT_RE.sub(_percent_to_words, text)
