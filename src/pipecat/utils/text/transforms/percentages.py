#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand percentage expressions into spoken form for TTS."""

import re

from num2words import num2words

from pipecat.frames.frames import TextType

_PERCENT_RE = re.compile(r"(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*%")

# A range such as "5-10%" or "5%–10%": the dash becomes "to". The first number must not
# be glued to a word ("Q3-5%"), so only standalone numbers start a range.
_PERCENT_RANGE_RE = re.compile(
    r"(?<![\w.,])(\d[\d,]*(?:\.\d+)?\s*%?)\s*[-–]\s*(?=\d[\d,]*(?:\.\d+)?\s*%)"
)

# A minus sign in front of a percentage, not a hyphen joined to a preceding word or number.
_NEGATIVE_PERCENT_RE = re.compile(r"(?<![\w.,%/+])[-−](?=\d[\d,]*(?:\.\d+)?\s*%)")


def _percent_to_words(match: re.Match) -> str:
    value = match.group(1).replace(",", "")
    if match.group(2):
        value += "." + match.group(2)
    number_str = num2words(float(value), lang="en")
    return f"{number_str} percent"


async def expand_percentages(text: str, text_type: str | TextType) -> str:
    """Expand percentage expressions to their spoken form.

    Args:
        text: Input text possibly containing percentage expressions.
        text_type: Text type of the text frame (unused).

    Returns:
        Text with percentages replaced by spoken equivalents.

    Example::

        result = await expand_percentages("50% off today", "*")
        # "fifty percent off today"
    """
    text = _PERCENT_RANGE_RE.sub(lambda m: f"{m.group(1).rstrip()} to ", text)
    text = _NEGATIVE_PERCENT_RE.sub("minus ", text)
    return _PERCENT_RE.sub(_percent_to_words, text)
