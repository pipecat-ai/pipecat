#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base class for network utilities, providing exponential backoff time calculation."""


def exponential_backoff_time(
    attempt: int, min_wait: float = 4, max_wait: float = 10, multiplier: float = 1
) -> float:
    """Calculate exponential backoff wait time.

    Args:
        attempt: Current attempt number (1-based)
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        multiplier: Base multiplier for exponential calculation

    Returns:
        Wait time in seconds
    """
    try:
        exp = 2 ** (attempt - 1) * multiplier
        result = max(0, min(exp, max_wait))
        return max(min_wait, result)
    except (ValueError, ArithmeticError):
        return max_wait
