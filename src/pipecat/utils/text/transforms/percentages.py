#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand percentage expressions into spoken form for TTS."""

import re

from num2words import num2words

from pipecat.frames.frames import AggregationType

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _percent_to_words(match: re.Match) -> str:
    value = match.group(1)
    number_str = num2words(float(value), lang="en")
    return f"{number_str} percent"


async def expand_percentages(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand percentage expressions to their spoken form.

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
