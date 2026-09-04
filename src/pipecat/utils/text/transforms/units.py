#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand unit abbreviations into their spoken form for TTS."""

import re

from pipecat.frames.frames import AggregationType

# Maps unit abbreviation to (singular, plural) spoken forms.
_UNIT_MAP: dict[str, tuple[str, str]] = {
    "km": ("kilometer", "kilometers"),
    "m": ("meter", "meters"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "mi": ("mile", "miles"),
    "ft": ("foot", "feet"),
    "in": ("inch", "inches"),
    "yd": ("yard", "yards"),
    "kg": ("kilogram", "kilograms"),
    "g": ("gram", "grams"),
    "mg": ("milligram", "milligrams"),
    "lb": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "l": ("liter", "liters"),
    "ml": ("milliliter", "milliliters"),
    "mph": ("mile per hour", "miles per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "kmh": ("kilometer per hour", "kilometers per hour"),
    "gb": ("gigabyte", "gigabytes"),
    "mb": ("megabyte", "megabytes"),
    "kb": ("kilobyte", "kilobytes"),
    "tb": ("terabyte", "terabytes"),
    "hz": ("hertz", "hertz"),
    "khz": ("kilohertz", "kilohertz"),
    "mhz": ("megahertz", "megahertz"),
    "ghz": ("gigahertz", "gigahertz"),
}

# Single-letter units that are also common English words: only expand when
# they appear immediately after a digit with no intervening space, e.g. "5m"
# but not "1 m people" (where "m" is a word) or "1 in 5" (preposition).
_AMBIGUOUS_UNITS = {"in", "m", "g", "l"}

_sorted_unambiguous = sorted(
    (u for u in _UNIT_MAP if u not in _AMBIGUOUS_UNITS), key=len, reverse=True
)
_sorted_ambiguous = sorted(_AMBIGUOUS_UNITS, key=len, reverse=True)

# Unambiguous units allow optional whitespace between the number and the unit.
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in _sorted_unambiguous) + r")\b",
    re.IGNORECASE,
)

# Ambiguous units require the unit to follow the digit with no space.
_AMBIGUOUS_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)(" + "|".join(re.escape(u) for u in _sorted_ambiguous) + r")\b",
    re.IGNORECASE,
)


async def expand_units(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand unit abbreviations to their full spoken form.

    A quantity of exactly one takes the singular form of the unit.

    Args:
        text: Input text possibly containing unit expressions.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with unit abbreviations replaced by spoken equivalents.

    Example::

        result = await expand_units("Run 5km at 100kph", "*")
        # "Run 5 kilometers at 100 kilometers per hour"

        result = await expand_units("Only 1km left", "*")
        # "Only 1 kilometer left"
    """

    def _replace(match: re.Match) -> str:
        number = match.group(1)
        singular, plural = _UNIT_MAP[match.group(2).lower()]
        # Only a bare "1" takes the singular; a decimal such as "1.0" reads as
        # plural in speech.
        return f"{number} {singular if number == '1' else plural}"

    text = _UNIT_RE.sub(_replace, text)
    return _AMBIGUOUS_UNIT_RE.sub(_replace, text)
