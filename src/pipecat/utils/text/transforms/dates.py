#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand date expressions into spoken form for TTS."""

import re
from datetime import datetime

from num2words import num2words

from pipecat.frames.frames import AggregationType

# ISO dates: 2023-05-10
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# US dates: 05/10/2023 or 05-10-2023 (not matched by ISO because ISO has 4-digit year first)
_US_DATE_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b")

_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    suffix = _ORDINAL_SUFFIXES.get(n % 10 if n % 100 not in (11, 12, 13) else 0, "th")
    return f"{n}{suffix}"


def _date_to_spoken(dt: datetime) -> str:
    year_words = num2words(dt.year, lang="en")
    month = dt.strftime("%B")
    day = _ordinal(dt.day)
    return f"{month} {day}, {year_words}"


def _iso_replace(match: re.Match) -> str:
    try:
        dt = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return _date_to_spoken(dt)
    except ValueError:
        return match.group(0)


def _us_replace(match: re.Match) -> str:
    try:
        dt = datetime(int(match.group(3)), int(match.group(1)), int(match.group(2)))
        return _date_to_spoken(dt)
    except ValueError:
        return match.group(0)


async def normalize_dates(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand date expressions to their spoken form.

    Handles ISO format (``YYYY-MM-DD``) and US format (``MM/DD/YYYY`` or ``MM-DD-YYYY``).

    Args:
        text: Input text possibly containing date expressions.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with date expressions replaced by spoken equivalents.

    Example::

        result = await normalize_dates("Meeting on 2023-05-10", "*")
        # "Meeting on May 10th, two thousand and twenty three"
    """
    text = _ISO_DATE_RE.sub(_iso_replace, text)
    text = _US_DATE_RE.sub(_us_replace, text)
    return text
