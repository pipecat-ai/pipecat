#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.utils.asyncio.task_manager import BaseTaskManager


class WatchdogQueue(asyncio.Queue):
    """An asynchronous queue that resets the current task watchdog timer. This
    is necessary to avoid task watchdog timers to expire while we are waiting to
    get an item from the queue.

    """

    def __init__(
        self,
        manager: BaseTaskManager,
        *,
        maxsize: int = 0,
        timeout: float = 2.0,
    ) -> None:
        super().__init__(maxsize)
        self._manager = manager
        self._timeout = timeout

    async def get(self):
        if self._manager.task_watchdog_enabled:
            return await self._watchdog_get()
        else:
            return await super().get()

    def task_done(self):
        if self._manager.task_watchdog_enabled:
            self._manager.task_reset_watchdog()
        super().task_done()

    async def _watchdog_get(self):
        while True:
            try:
                item = await asyncio.wait_for(super().get(), timeout=self._timeout)
                self._manager.task_reset_watchdog()
                return item
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()
