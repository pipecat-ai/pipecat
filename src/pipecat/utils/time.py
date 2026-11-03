#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Time utilities for the Pipecat framework.

This module provides utility functions for time handling including
ISO8601 formatting, nanosecond conversions, and human-readable
time string formatting.
"""

import datetime


def time_now_iso8601() -> str:
    """Get the current UTC time as an ISO8601 formatted string.

    Returns:
        The current UTC time in ISO8601 format with millisecond precision.
    """
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")


def seconds_to_nanoseconds(seconds: float) -> int:
    """Convert seconds to nanoseconds.

    Args:
        seconds: The number of seconds to convert.

    Returns:
        The equivalent number of nanoseconds as an integer.
    """
    return int(seconds * 1_000_000_000)


def nanoseconds_to_seconds(nanoseconds: int) -> float:
    """Convert nanoseconds to seconds.

    Args:
        nanoseconds: The number of nanoseconds to convert.

    Returns:
        The equivalent number of seconds as a float.
    """
    return nanoseconds / 1_000_000_000


def nanoseconds_to_str(nanoseconds: int) -> str:
    """Convert nanoseconds to a human-readable time string.

    Args:
        nanoseconds: The number of nanoseconds to convert.

    Returns:
        A formatted time string in "H:MM:SS.microseconds" format.
    """
    # Integer arithmetic: float seconds can't represent most microsecond values exactly.
    total_microseconds = int(nanoseconds) // 1_000
    hours, remainder = divmod(total_microseconds, 3_600_000_000)
    minutes, remainder = divmod(remainder, 60_000_000)
    seconds, microseconds = divmod(remainder, 1_000_000)
    return f"{hours}:{minutes:02}:{seconds:02}.{microseconds:06}"
