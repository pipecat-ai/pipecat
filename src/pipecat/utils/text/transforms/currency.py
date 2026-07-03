#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand currency amounts into spoken form for TTS."""

import re

from num2words import num2words

from pipecat.frames.frames import AggregationType

# Maps currency symbol to (singular, plural, cents_singular, cents_plural)
_CURRENCY_MAP: dict[str, tuple[str, str, str | None, str | None]] = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "¥": ("yen", "yen", None, None),
    "₹": ("rupee", "rupees", "paisa", "paise"),
}

_CURRENCY_RE = re.compile(r"([€£¥₹\$])\s*(\d{1,3}(?:,\d{3})*|\d+)\b(?:\.(\d{1,2}))?")


def _amount_to_words(n: float, singular: str, plural: str) -> str:
    words = num2words(n, lang="en")
    unit = singular if n == 1 else plural
    return f"{words} {unit}"


def _currency_match(match: re.Match) -> str:
    symbol = match.group(1)
    whole_str = match.group(2).replace(",", "")
    frac_str = match.group(3)

    currency = _CURRENCY_MAP.get(symbol, ("unit", "units", "cent", "cents"))
    singular, plural, c_singular, c_plural = currency

    whole = int(whole_str)
    result = _amount_to_words(whole, singular, plural)

    if frac_str and c_singular and c_plural:
        frac = int(frac_str.ljust(2, "0"))
        if frac > 0:
            result += " and " + _amount_to_words(frac, c_singular, c_plural)

    return result


async def expand_currency(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand currency amounts to their spoken form.

    Args:
        text: Input text possibly containing currency expressions.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with currency amounts replaced by spoken equivalents.

    Example::

        result = await expand_currency("Your balance is $42.50", "*")
        # "Your balance is forty-two dollars and fifty cents"
    """
    return _CURRENCY_RE.sub(_currency_match, text)
