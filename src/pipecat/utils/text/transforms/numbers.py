#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand numeric expressions into spoken form for TTS."""

import re
from collections.abc import Callable

from num2words import num2words

from pipecat.frames.frames import AggregationType

# Matches integers and decimals, with optional thousand separators.
_NUMBER_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\b")


def expand_numbers(
    digit_cutoff: int | None = 2025,
) -> Callable[[str, str | AggregationType], object]:
    """Return a transform that expands numbers to their spoken form.

    When *digit_cutoff* is set, numbers above it are read digit-by-digit
    (e.g. ``"2026"`` → ``"2 0 2 6"``). Numbers at or below the cutoff are
    expanded as quantities (e.g. ``"42"`` → ``"forty two"``). Pass ``None``
    to expand all numbers as words regardless of magnitude.

    Args:
        digit_cutoff: Numbers larger than this value are read digit-by-digit.
            ``None`` disables the cutoff so every number is expanded as a word.

    Returns:
        An async transform callable compatible with ``text_transforms``.

    Example::

        transform = expand_numbers(digit_cutoff=2025)
        result = await transform("Room 42 has 1234 seats and opens in 2026", "*")
        # "Room forty two has one thousand two hundred thirty four seats and opens in 2 0 2 6"
    """

    def _num_to_words(match: re.Match) -> str:
        whole_str = match.group(1).replace(",", "")
        frac_str = match.group(2)
        whole = int(whole_str)

        if digit_cutoff is not None and whole > digit_cutoff:
            result = " ".join(whole_str)
            if frac_str:
                result += " point " + " ".join(frac_str)
            return result

        if frac_str:
            words = num2words(float(f"{whole_str}.{frac_str}"), lang="en")
        else:
            words = num2words(whole, lang="en")

        return words

    async def _transform(text: str, aggregation_type: str | AggregationType) -> str:
        return _NUMBER_RE.sub(_num_to_words, text)

    return _transform
