#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""System clock implementation for Pipecat."""

import time

from pipecat.clocks.base_clock import BaseClock


class SystemClock(BaseClock):
    """A monotonic clock implementation using system time.

    Provides high-precision timing using the system's monotonic clock,
    which is not affected by system clock adjustments and is suitable
    for measuring elapsed time in real-time applications.
    """

    def __init__(self):
        """Initialize the system clock.

        The clock starts in an uninitialized state and must be started
        explicitly using the start() method before time measurement begins.
        """
        self._time = 0

    def get_time(self) -> int:
        """Get the elapsed time since the clock was started.

        Returns:
            The elapsed time in nanoseconds since start() was called.
            Returns 0 if the clock has not been started yet.
        """
        return time.monotonic_ns() - self._time if self._time > 0 else 0

    def start(self):
        """Start the clock and begin time measurement.

        Records the current monotonic time as the reference point
        for all subsequent get_time() calls.
        """
        self._time = time.monotonic_ns()
