#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand unit abbreviations into their spoken form for TTS."""

import re

from pipecat.frames.frames import AggregationType

_UNIT_MAP: dict[str, str] = {
    "km": "kilometers",
    "m": "meters",
    "cm": "centimeters",
    "mm": "millimeters",
    "mi": "miles",
    "ft": "feet",
    "in": "inches",
    "yd": "yards",
    "kg": "kilograms",
    "g": "grams",
    "mg": "milligrams",
    "lb": "pounds",
    "oz": "ounces",
    "l": "liters",
    "ml": "milliliters",
    "mph": "miles per hour",
    "kph": "kilometers per hour",
    "kmh": "kilometers per hour",
    "gb": "gigabytes",
    "mb": "megabytes",
    "kb": "kilobytes",
    "tb": "terabytes",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
}

# Build pattern from longest to shortest to avoid partial matches
_sorted_units = sorted(_UNIT_MAP.keys(), key=len, reverse=True)
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in _sorted_units) + r")\b",
    re.IGNORECASE,
)


async def expand_units(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand unit abbreviations to their full spoken form.

    Args:
        text: Input text possibly containing unit expressions.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with unit abbreviations replaced by spoken equivalents.

    Example::

        result = await expand_units("Run 5km at 100kph", "*")
        # "Run 5 kilometers at 100 kilometers per hour"
    """

    def _replace(match: re.Match) -> str:
        number = match.group(1)
        unit = _UNIT_MAP[match.group(2).lower()]
        return f"{number} {unit}"

    return _UNIT_RE.sub(_replace, text)
