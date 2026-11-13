#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand currency amounts into spoken form for TTS."""

import re

from num2words import num2words

from pipecat.frames.frames import TextType

# Maps currency symbol to (singular, plural, cents_singular, cents_plural)
_CURRENCY_MAP: dict[str, tuple[str, str, str | None, str | None]] = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "¥": ("yen", "yen", None, None),
    "₹": ("rupee", "rupees", "paisa", "paise"),
}

# Abbreviated scales written directly after an amount ("$5k", "$2.5m", "$1.5bn").
_SCALE_ABBREVIATIONS: dict[str, str] = {
    "k": "thousand",
    "m": "million",
    "mn": "million",
    "b": "billion",
    "bn": "billion",
    "t": "trillion",
    "tn": "trillion",
}

# An amount may be followed by a scale word ("$2.5 billion") or its abbreviation
# ("$2.5bn"), which multiplies the whole amount, so the fraction is part of the
# number rather than a subunit. The amount must end at a word boundary, so an
# unrecognised suffix ("$1.5x") is left alone instead of being read as cents.
_CURRENCY_RE = re.compile(
    r"([€£¥₹\$])\s*(\d{1,3}(?:,\d{3})*|\d+)(?!\d)(?:\.(\d+))?(?!\.\d)"
    r"(?:\s+(thousand|million|billion|trillion)\b|(k|mn|m|bn|b|tn|t)\b)?(?!\w)",
    re.IGNORECASE,
)

# A minus sign in front of a currency amount, not a hyphen in a range like $10-$20.
_NEGATIVE_CURRENCY_RE = re.compile(r"(?<![\w.,])[-−](?=\s*[€£¥₹\$])")


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

    scale = match.group(4)
    if not scale and match.group(5):
        scale = _SCALE_ABBREVIATIONS[match.group(5).lower()]
    if scale:
        # "$2.5 billion" and "$2.5bn" read "two point five billion dollars": the fraction
        # belongs to the scaled number, and the unit is always plural.
        number = num2words(whole, lang="en")
        fraction = frac_str.rstrip("0") if frac_str else ""
        if fraction:
            # "$2.50 million" is two and a half million, so a trailing zero in
            # the written fraction is not a digit to read out.
            number += " point " + " ".join(num2words(int(d), lang="en") for d in fraction)
        return f"{number} {scale.lower()} {plural}"

    result = _amount_to_words(whole, singular, plural)

    if frac_str and c_singular and c_plural:
        frac = int(frac_str[:2].ljust(2, "0"))
        if frac > 0:
            result += " and " + _amount_to_words(frac, c_singular, c_plural)

    return result


def _keep_code_apart(match: re.Match) -> str:
    """Expand *match*, keeping a currency code written directly before the symbol separate.

    ``US$5`` otherwise becomes ``USfive dollars``: the substitution replaces the
    symbol, so the code and the spoken amount end up glued together.
    """
    spoken = _currency_match(match)
    if match.start() and match.string[match.start() - 1].isalpha():
        return " " + spoken
    return spoken


async def expand_currency(text: str, text_type: str | TextType) -> str:
    """Expand currency amounts to their spoken form.

    Args:
        text: Input text possibly containing currency expressions.
        text_type: Text type of the text frame (unused).

    Returns:
        Text with currency amounts replaced by spoken equivalents.

    Example::

        result = await expand_currency("Your balance is $42.50", "*")
        # "Your balance is forty-two dollars and fifty cents"
    """
    text = _NEGATIVE_CURRENCY_RE.sub("minus ", text)
    return _CURRENCY_RE.sub(_keep_code_apart, text)
