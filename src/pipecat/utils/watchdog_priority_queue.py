#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.utils.watchdog_reseter import WatchdogReseter


class WatchdogPriorityQueue(asyncio.PriorityQueue):
    """An asynchronous priority queue that resets the current task watchdog
    timer. This is necessary to avoid task watchdog timers to expire while we
    are waiting to get an item from the queue.

    """

    def __init__(
        self,
        reseter: WatchdogReseter,
        *,
        maxsize: int = 0,
        timeout: float = 2.0,
        watchdog_enabled: bool = False,
    ) -> None:
        super().__init__(maxsize)
        self._reseter = reseter
        self._timeout = timeout
        self._watchdog_enabled = watchdog_enabled

    async def get(self):
        if self._watchdog_enabled:
            return await self._watchdog_get()
        else:
            return await super().get()

    def task_done(self):
        if self._watchdog_enabled:
            self._reseter.reset_watchdog()
        super().task_done()

    async def _watchdog_get(self):
        while True:
            try:
                item = await asyncio.wait_for(super().get(), timeout=self._timeout)
                self._reseter.reset_watchdog()
                return item
            except asyncio.TimeoutError:
                self._reseter.reset_watchdog()
