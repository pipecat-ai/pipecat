#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand unit abbreviations into their spoken form for TTS."""

import re

from pipecat.frames.frames import TextType

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

# Maps a time unit written after a slash to its spoken form, making the unit
# before the slash a rate: "km/h", "m/s", "MB/s".
_PER_UNIT_MAP: dict[str, str] = {
    "h": "hour",
    "hr": "hour",
    "hour": "hour",
    "s": "second",
    "sec": "second",
    "second": "second",
    "min": "minute",
    "minute": "minute",
}

# Single-letter units that are also common English words: only expand when
# they appear immediately after a digit with no intervening space, e.g. "5m"
# but not "1 m people" (where "m" is a word) or "1 in 5" (preposition).
_AMBIGUOUS_UNITS = {"in", "m", "g", "l"}

_sorted_unambiguous = sorted(
    (u for u in _UNIT_MAP if u not in _AMBIGUOUS_UNITS), key=len, reverse=True
)
# Liters are also written with a capital "L".
_sorted_ambiguous = sorted(_AMBIGUOUS_UNITS | {"L"}, key=len, reverse=True)

# A rate expands whole, so it is matched before single units. The slash marks
# any unit as one, so even an ambiguous unit may follow a space ("9.8 m/s").
# Optional spaces around the slash cover the common written form "km / h".
_RATE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*("
    + "|".join(re.escape(u) for u in sorted(_UNIT_MAP, key=len, reverse=True))
    + r")\s*/\s*("
    + "|".join(re.escape(u) for u in sorted(_PER_UNIT_MAP, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# Unambiguous units allow optional whitespace between the number and the unit.
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in _sorted_unambiguous) + r")\b",
    re.IGNORECASE,
)

# Ambiguous units require the unit to follow the digit with no space, and match
# case-sensitively: "5M" (million) and "5G" (the network) are not meters or grams.
_AMBIGUOUS_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)(" + "|".join(re.escape(u) for u in _sorted_ambiguous) + r")\b"
)

# Degrees: "25°C", "72 °F", "a 90° turn". The degree sign is not a word, so a TTS
# engine may skip it or read the scale letter on its own ("twenty-five C").
_DEGREE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*°(?:\s*([CF])(?![a-zA-Z])|(?![a-zA-Z\d]))")
_DEGREE_SCALES = {"C": " Celsius", "F": " Fahrenheit"}


async def expand_units(text: str, text_type: str | TextType) -> str:
    """Expand unit abbreviations to their full spoken form.

    A quantity of exactly one takes the singular form of the unit.

    Args:
        text: Input text possibly containing unit expressions.
        text_type: Text type of the text frame (unused).

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

    def _replace_degrees(match: re.Match) -> str:
        number = match.group(1)
        degrees = "degree" if number == "1" else "degrees"
        return f"{number} {degrees}{_DEGREE_SCALES.get(match.group(2) or '', '')}"

    def _replace_rate(match: re.Match) -> str:
        return f"{_replace(match)} per {_PER_UNIT_MAP[match.group(3).lower()]}"

    text = _DEGREE_RE.sub(_replace_degrees, text)
    text = _RATE_RE.sub(_replace_rate, text)
    text = _UNIT_RE.sub(_replace, text)
    return _AMBIGUOUS_UNIT_RE.sub(_replace, text)
