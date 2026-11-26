#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared handling of a leading minus sign for the number-like transforms."""

import re


def minus_before(amount_start: str) -> re.Pattern[str]:
    """Build a pattern for a minus sign touching an amount.

    The sign must not follow a word character, ``.`` or ``,``, so a hyphen
    inside "5-10", "555-1234" or "COVID-19" is not a minus. A spaced dash such
    as "10 - 5" does not touch the amount, so it is not matched either.

    Args:
        amount_start: Regex fragment for the first character of the amount.

    Returns:
        A compiled pattern matching only the sign (hyphen or Unicode minus).
    """
    return re.compile(rf"(?<![\w.,])[-−](?={amount_start})")


def say_minus(pattern: re.Pattern[str], text: str) -> str:
    """Replace each sign matched by ``pattern`` with the word "minus "."""
    return pattern.sub("minus ", text)
