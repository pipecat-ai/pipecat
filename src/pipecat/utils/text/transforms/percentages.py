#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand percentage expressions into spoken form for TTS."""

import re
from collections.abc import Awaitable
from typing import overload

from num2words import num2words

from pipecat.frames.frames import TextType
from pipecat.utils.deprecation import warn_deprecated
from pipecat.utils.text.transforms.text_transform import TextTransform

# Matches integers and decimals, with optional thousand separators.
_PERCENT_RE = re.compile(r"(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*%")


@overload
def expand_percentages(*, read_trailing_zeros: bool = False) -> TextTransform: ...


@overload
def expand_percentages(text: str, text_type: str | TextType) -> Awaitable[str]: ...


def expand_percentages(
    text: str | None = None,
    text_type: str | TextType | None = None,
    *,
    read_trailing_zeros: bool = False,
) -> TextTransform | Awaitable[str]:
    """Return a transform that expands percentage expressions to their spoken form.

    By default a decimal is read as a quantity, so trailing zeros are dropped
    (``"1.0%"`` becomes ``"one percent"``). With *read_trailing_zeros* set, the
    fraction is read one digit at a time, like ``expand_numbers``, so every
    written digit is spoken (``"1.10%"`` becomes ``"one point one zero percent"``).

    Args:
        text: Text to transform directly, when ``expand_percentages`` itself is
            used as the transform.

            .. deprecated:: 1.18.0
                Use ``expand_percentages()`` instead, which returns the transform.
                Will be removed in 2.0.0.

        text_type: Text type of the text frame, with *text*.

            .. deprecated:: 1.18.0
                Use ``expand_percentages()`` instead, which returns the transform.
                Will be removed in 2.0.0.

        read_trailing_zeros: Read every written fraction digit, including
            trailing zeros. Defaults to ``False``.

    Returns:
        An async transform callable compatible with ``text_transforms``.

    Example::

        transform = expand_percentages()
        result = await transform("50% off today", "*")
        # "fifty percent off today"
    """
    if text is not None:
        warn_deprecated(
            "`expand_percentages(text, text_type)` is deprecated since 1.18.0 and will be "
            "removed in 2.0.0. Use `expand_percentages()` instead.",
            stacklevel=2,
        )
        return _percent_transform(read_trailing_zeros=False)(text, text_type or "")
    return _percent_transform(read_trailing_zeros=read_trailing_zeros)


def _percent_transform(read_trailing_zeros: bool) -> TextTransform:
    def _percent_to_words(match: re.Match) -> str:
        whole_str = match.group(1).replace(",", "")
        frac_str = match.group(2)
        if frac_str and read_trailing_zeros:
            number_str = f"{num2words(int(whole_str), lang='en')} point " + " ".join(
                num2words(int(digit), lang="en") for digit in frac_str
            )
        else:
            value = f"{whole_str}.{frac_str}" if frac_str else whole_str
            number_str = num2words(float(value), lang="en")
        return f"{number_str} percent"

    async def _transform(text: str, text_type: str | TextType) -> str:
        return _PERCENT_RE.sub(_percent_to_words, text)

    return _transform
