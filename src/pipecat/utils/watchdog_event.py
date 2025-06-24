#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.utils.watchdog_reseter import WatchdogReseter


class WatchdogEvent(asyncio.Event):
    """An asynchronous event that resets the current task watchdog timer. This
    is necessary to avoid task watchdog timers to expire while we are waiting on
    the event.

    """

    def __init__(self, reseter: WatchdogReseter, timeout: float = 2.0) -> None:
        super().__init__()
        self._reseter = reseter
        self._timeout = timeout

    async def wait(self):
        while True:
            try:
                await asyncio.wait_for(super().wait(), timeout=self._timeout)
                self._reseter.reset_watchdog()
                return True
            except asyncio.TimeoutError:
                self._reseter.reset_watchdog()
