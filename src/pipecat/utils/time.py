#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import datetime


def time_now_iso8601() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def seconds_to_nanoseconds(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


def nanoseconds_to_seconds(nanoseconds: int) -> float:
    return nanoseconds / 1_000_000_000


def nanoseconds_to_str(nanoseconds: int) -> str:
    total_seconds = nanoseconds_to_seconds(nanoseconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    microseconds = int((total_seconds - int(total_seconds)) * 1_000_000)
    return f"{hours}:{minutes:02}:{seconds:02}.{microseconds:06}"
