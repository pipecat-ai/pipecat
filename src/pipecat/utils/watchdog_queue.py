#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.utils.watchdog_reseter import WatchdogReseter


class WatchdogQueue(asyncio.Queue):
    """An asynchronous queue that resets the current task watchdog timer. This
    is necessary to avoid task watchdog timers to expire while we are waiting to
    get an item from the queue.

    """

    def __init__(self, reseter: WatchdogReseter, maxsize: int = 0, timeout: float = 2.0) -> None:
        super().__init__(maxsize)
        self._reseter = reseter
        self._timeout = timeout

    async def get(self):
        while True:
            try:
                item = await asyncio.wait_for(super().get(), timeout=self._timeout)
                self._reseter.reset_watchdog()
                return item
            except asyncio.TimeoutError:
                self._reseter.reset_watchdog()

    def task_done(self):
        self._reseter.reset_watchdog()
        super().task_done()
