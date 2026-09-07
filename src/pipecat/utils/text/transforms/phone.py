#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expand phone numbers into space-separated digit sequences for TTS."""

import re

from pipecat.frames.frames import AggregationType

# Matches common phone formats: (123) 456-7890, 123-456-7890, 123.456.7890,
# +1 800 555 1234, etc.
_PHONE_RE = re.compile(
    r"(?<!\d)"  # not preceded by a digit (no partial match inside a longer number)
    r"(\+?1[\s.\-]?)?"  # optional country code
    r"(\(?\d{3}\)?[\s.\-]?)"  # area code
    r"(\d{3}[\s.\-]?)"  # exchange
    r"(\d{4})"  # subscriber
    r"(?!\d)"  # not followed by a digit
)


def _space_digits(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return " ".join(digits)


async def expand_phone_numbers(text: str, aggregation_type: str | AggregationType) -> str:
    """Space out phone number digits so TTS reads them individually.

    This transformer is alphanumeric-preserving: digits are kept, only separators
    change to spaces, so the :class:`WordCompletionTracker` requires no segment-map
    overhead.

    Args:
        text: Input text possibly containing phone numbers.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with phone numbers replaced by space-separated digit sequences.

    Example::

        result = await expand_phone_numbers("Call 123-456-7890 now", "*")
        # "Call 1 2 3 4 5 6 7 8 9 0 now"
    """
    return _PHONE_RE.sub(_space_digits, text)
