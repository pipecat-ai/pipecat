#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time

from pipecat.clocks.base_clock import BaseClock


class SystemClock(BaseClock):
    def __init__(self):
        self._time = 0

    def get_time(self) -> int:
        return time.monotonic_ns() - self._time if self._time > 0 else 0

    def start(self):
        self._time = time.monotonic_ns()
